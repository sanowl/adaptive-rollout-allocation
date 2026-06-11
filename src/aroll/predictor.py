"""Learned rollout predictor — replaces the paper's Gaussian process (Rec. 1).

The VIP paper predicts per-prompt success probability with a Gaussian process
over prompt embeddings and recursive Bayesian posterior updates.  That is elegant
but scales poorly (an O(Q^2) kernel matrix, cubic posterior updates).

Here we use a *tiny MLP value head* over prompt embeddings (or hidden states)
that predicts three quantities per prompt::

    p_success(prompt)            in (0, 1)
    uncertainty(prompt)          >= 0   (epistemic, via MC-dropout)
    expected_training_gain(prompt)

These feed the allocation either through the paper coefficient
``a_q = 4 sigma_Z^2 p(1-p)`` or through the boundary-seeking score
``p (1 - p) * uncertainty`` (see :mod:`aroll.scoring`).

The predictor is updated online from observed rollout success rates, mirroring
the GP's recursive update but as a couple of gradient steps — so it tracks the
non-stationary success probabilities as the policy improves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Prediction:
    """Per-prompt predictor outputs (numpy arrays, shape ``(B,)``)."""

    p_success: np.ndarray
    uncertainty: np.ndarray
    expected_training_gain: np.ndarray


class RolloutPredictor(nn.Module):
    """Tiny shared-trunk MLP with three heads + MC-dropout uncertainty.

    Args:
        embed_dim: dimensionality of prompt embeddings / hidden states.
        hidden: trunk width.
        dropout: MC-dropout rate (kept active at predict time for uncertainty).
        lr: Adam learning rate for online updates.
    """

    def __init__(self, embed_dim: int, hidden: int = 128, dropout: float = 0.1,
                 lr: float = 1e-3, device: str | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.p_head = nn.Linear(hidden, 1)    # logit of success probability
        self.etg_head = nn.Linear(hidden, 1)  # expected training gain (>=0 via softplus)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    # -- forward -------------------------------------------------------------
    def _forward_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.p_head(h).squeeze(-1), F.softplus(self.etg_head(h).squeeze(-1))

    @torch.no_grad()
    def predict(self, embeddings: np.ndarray, mc_samples: int = 8) -> Prediction:
        """Predict p_success / uncertainty / ETG with MC-dropout.

        ``uncertainty`` is the std of ``p_success`` across ``mc_samples`` dropout
        masks — high where the model is unsure (sparsely observed prompts).
        """
        x = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32, device=self.device)
        if mc_samples > 1:
            self.train()  # enable dropout
            ps, etgs = [], []
            for _ in range(mc_samples):
                logit, etg = self._forward_logits(x)
                ps.append(torch.sigmoid(logit))
                etgs.append(etg)
            P = torch.stack(ps, 0)            # (S, B)
            p_mean = P.mean(0)
            uncertainty = P.std(0, unbiased=False)
            etg = torch.stack(etgs, 0).mean(0)
        else:
            self.eval()
            logit, etg = self._forward_logits(x)
            p_mean = torch.sigmoid(logit)
            uncertainty = torch.zeros_like(p_mean)
        self.eval()
        return Prediction(
            p_success=p_mean.cpu().numpy(),
            uncertainty=uncertainty.cpu().numpy(),
            expected_training_gain=etg.cpu().numpy(),
        )

    # -- online update -------------------------------------------------------
    def update(
        self,
        embeddings: np.ndarray,
        successes: np.ndarray,
        counts: np.ndarray,
        sample_weight: np.ndarray | None = None,
        etg_target: np.ndarray | None = None,
        steps: int = 5,
    ) -> float:
        """Online update from observed rollouts (the GP's recursive analogue).

        Args:
            embeddings: prompt embeddings, shape ``(B, d)``.
            successes: number of successful rollouts per prompt, shape ``(B,)``.
            counts: total rollouts per prompt, shape ``(B,)`` (>= 1).
            sample_weight: optional per-prompt weight (e.g. replay staleness).
            etg_target: regression target for the expected-training-gain head.
                :class:`~aroll.vip.VIPAllocator` passes a *real* signal here —
                learning progress, the rise in success rate since the prompt was
                last seen. If omitted it falls back to the boundary proxy
                ``p_hat (1 - p_hat)`` (note: that proxy is circular with the
                allocation coefficient, so prefer passing a real target).
            steps: gradient steps per update.

        Returns:
            final loss value.
        """
        x = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32, device=self.device)
        succ = torch.as_tensor(np.asarray(successes), dtype=torch.float32, device=self.device)
        cnt = torch.as_tensor(np.asarray(counts), dtype=torch.float32, device=self.device).clamp_min(1.0)
        p_hat = (succ / cnt).clamp(1e-4, 1 - 1e-4)
        if sample_weight is None:
            w = torch.ones_like(cnt)
        else:
            w = torch.as_tensor(np.asarray(sample_weight), dtype=torch.float32, device=self.device)
        # Weight observations by rollout count (more rollouts => more reliable).
        w = w * cnt
        denom = w.sum().clamp_min(1e-8)        # robust to all-zero bagging weights
        if etg_target is None:
            etg_t = p_hat * (1.0 - p_hat)
        else:
            etg_t = torch.as_tensor(np.asarray(etg_target), dtype=torch.float32, device=self.device)

        self.train()
        loss_val = 0.0
        for _ in range(steps):
            self.opt.zero_grad()
            logit, etg = self._forward_logits(x)
            # Binomial NLL on success counts, weighted by reliability.
            bce = F.binary_cross_entropy_with_logits(logit, p_hat, reduction="none")
            loss_p = (w * bce).sum() / denom
            loss_etg = (w * (etg - etg_t) ** 2).sum() / denom
            loss = loss_p + 0.5 * loss_etg
            loss.backward()
            self.opt.step()
            loss_val = float(loss.detach())
        self.eval()
        return loss_val


class EnsemblePredictor:
    """Deep ensemble of :class:`RolloutPredictor` members for *calibrated*
    epistemic uncertainty (Review fix #3 — replaces MC-dropout).

    Uncertainty is the disagreement (std) of ``p_success`` across independently
    initialised members. Deep ensembles are the standard strong baseline for
    calibrated uncertainty and degrade gracefully as a prompt accrues evidence:
    members converge -> disagreement shrinks. Members are decorrelated by (i)
    different random initialisation and (ii) *online bagging* — each member
    weights every observation by an independent ``Poisson(1)`` draw, the
    streaming analogue of a bootstrap resample.

    Drop-in compatible with :class:`RolloutPredictor`: same ``predict`` / ``update``
    signatures, so it can be passed straight to :class:`~aroll.vip.VIPAllocator`.
    """

    def __init__(self, embed_dim: int, n_members: int = 5, hidden: int = 128,
                 dropout: float = 0.0, lr: float = 1e-3, device: str | None = None,
                 seed: int = 0):
        self.embed_dim = embed_dim
        self.n_members = n_members
        self._rng = np.random.default_rng(seed)
        self.members: list[RolloutPredictor] = []
        for k in range(n_members):
            torch.manual_seed(seed + 1000 * (k + 1))   # diverse initialisation
            self.members.append(RolloutPredictor(embed_dim, hidden, dropout, lr, device))

    @torch.no_grad()
    def predict(self, embeddings: np.ndarray, mc_samples: int = 1) -> Prediction:
        # Each member does one deterministic forward (dropout off); the ensemble
        # spread is the uncertainty. ``mc_samples`` is accepted for interface
        # parity but ignored (ensemble disagreement supplies the uncertainty).
        ps, etgs = [], []
        for m in self.members:
            pr = m.predict(embeddings, mc_samples=1)
            ps.append(pr.p_success)
            etgs.append(pr.expected_training_gain)
        P = np.stack(ps, 0)                              # (M, B)
        return Prediction(
            p_success=P.mean(0),
            uncertainty=P.std(0),                        # epistemic disagreement
            expected_training_gain=np.stack(etgs, 0).mean(0),
        )

    def update(self, embeddings: np.ndarray, successes: np.ndarray, counts: np.ndarray,
               sample_weight: np.ndarray | None = None, etg_target: np.ndarray | None = None,
               steps: int = 5) -> float:
        B = np.asarray(successes).shape[0]
        base = np.ones(B) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        losses = []
        for m in self.members:
            poisson = self._rng.poisson(1.0, size=B).astype(float)   # online bagging
            w = base * poisson
            losses.append(m.update(embeddings, successes, counts, sample_weight=w,
                                   etg_target=etg_target, steps=steps))
        return float(np.mean(losses))


class EMAPredictor:
    """Lightweight per-prompt EMA baseline (no torch) — analogous to the paper's
    Moving-Average ablation.  Useful for tests and as a sanity baseline."""

    def __init__(self, num_prompts: int, decay: float = 0.7, prior: float = 0.5):
        self.p = np.full(num_prompts, prior, dtype=float)
        self.seen = np.zeros(num_prompts, dtype=int)
        self.decay = decay

    def predict(self, prompt_ids: np.ndarray) -> Prediction:
        p = self.p[prompt_ids]
        # Uncertainty shrinks as a prompt is observed more often.
        unc = 1.0 / np.sqrt(1.0 + self.seen[prompt_ids])
        return Prediction(p_success=p, uncertainty=unc, expected_training_gain=p * (1 - p))

    def update(self, prompt_ids: np.ndarray, successes: np.ndarray, counts: np.ndarray) -> None:
        p_hat = np.clip(successes / np.maximum(counts, 1), 0.0, 1.0)
        ids = np.asarray(prompt_ids)
        self.p[ids] = self.decay * self.p[ids] + (1 - self.decay) * p_hat
        self.seen[ids] += 1
