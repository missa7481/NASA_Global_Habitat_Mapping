"""Observation-aware deep SDM: shared latent suitability + source observation models.

Notation
--------
h(x)  habitat representation (environment and/or satellite embedding)
e(x)  effort / human-access covariates seen by the reporting-bias head
eta   = f_theta(h)                      latent log-odds habitat suitability (shared)

Opportunistic presence vs. target-group background (thinned point processes):
    logit P(target | reported culicid at x) = eta(x) + g_phi(e(x)) + c_po
Survey source s, site i, sampling event j (repeated-visit occupancy model):
    z_i ~ Bern(sigmoid(eta_i + alpha_s))
    y_ij | z_i=1 ~ Bern(sigmoid(delta_s + kappa_s * log_effort_ij)),   y_ij | z_i=0 = 0
The survey likelihood marginalises z_i, so a site with zero detections over few
low-effort visits is weak evidence of unsuitability, and the occupancy part is
separated from per-visit detectability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    name: str
    use_po: bool = True
    use_survey: bool = True
    bias_head: bool = True
    pooled: bool = False          # naive baseline: survey sites as binary labels, no detection model
    linear: bool = False          # linear eta (classical ISDM reference)
    orth_e: int = 0               # >0: last orth_e columns of h are effort covariates e; the rest is the
                                  # effort-orthogonalised representation h_perp. eta = f(h_perp) + m(e), m linear.
    hidden: int = 64
    bias_hidden: int = 16
    dropout: float = 0.1
    lr: float = 3e-3
    weight_decay: float = 0.0       # unused (kept for config compatibility)
    l2: float = 1e-3                # explicit L2 on f/g weights; tuned by inner CV when --tune
    epochs: int = 400


def mlp(d_in: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout)]
        d = hidden
    mods.append(nn.Linear(d, 1))
    return nn.Sequential(*mods)


class OADSDM(nn.Module):
    def __init__(self, cfg: Config, d_h: int, d_e: int, n_survey: int):
        super().__init__()
        self.cfg = cfg
        d_f = d_h - cfg.orth_e
        self.f = nn.Linear(d_f, 1) if cfg.linear else mlp(d_f, cfg.hidden, 2, cfg.dropout)
        self.m = nn.Linear(cfg.orth_e, 1, bias=False) if cfg.orth_e else None
        self.g = (nn.Linear(d_e, 1) if cfg.linear else mlp(d_e, cfg.bias_hidden, 1, 0.0)) if cfg.bias_head else None
        self.c_po = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(n_survey))
        self.delta = nn.Parameter(torch.full((n_survey,), -2.0))
        self.kappa = nn.Parameter(torch.zeros(n_survey))

    def eta(self, h: torch.Tensor) -> torch.Tensor:
        if self.m is None:
            return self.f(h).squeeze(-1)
        k = self.cfg.orth_e
        return (self.f(h[:, :-k]) + self.m(h[:, -k:])).squeeze(-1)

    # ---------------- likelihood terms ----------------
    def nll_po(self, h, e, y):
        logit = self.eta(h) + self.c_po
        if self.g is not None:
            logit = logit + self.g(e).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logit, y)

    def nll_survey(self, h, src, ev_site, ev_logeff, ev_pos, ev_neg, has_det):
        """Per-source mean negative log marginal likelihood of detection histories."""
        n_sites = h.shape[0]
        psi_logit = self.eta(h) + self.alpha[src]
        ev_src = src[ev_site]
        p_logit = self.delta[ev_src] + self.kappa[ev_src] * ev_logeff
        ll_ev = ev_pos * F.logsigmoid(p_logit) + ev_neg * F.logsigmoid(-p_logit)
        ll_hist_z1 = torch.zeros(n_sites, device=h.device).index_add(0, ev_site, ll_ev)
        ll_occ = F.logsigmoid(psi_logit) + ll_hist_z1
        ll_site = torch.where(has_det, ll_occ, torch.logaddexp(ll_occ, F.logsigmoid(-psi_logit)))
        losses = [-(ll_site[src == s]).mean() for s in torch.unique(src)]
        return torch.stack(losses).sum()

    def nll_pooled_survey(self, h, src, has_det):
        logit = self.eta(h) + self.alpha[src]
        y = has_det.float()
        losses = [F.binary_cross_entropy_with_logits(logit[src == s], y[src == s]) for s in torch.unique(src)]
        return torch.stack(losses).sum()


def to_tensors(batch: dict, device: str) -> dict:
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items() if k != "n_survey"}


def objective(model: OADSDM, cfg: Config, t: dict) -> torch.Tensor:
    """The model's own data objective: sum of per-source mean negative log-likelihoods."""
    loss = torch.zeros((), device=t["po_h"].device)
    if cfg.use_po and len(t["po_y"]):
        loss = loss + model.nll_po(t["po_h"], t["po_e"], t["po_y"])
    if cfg.use_survey and len(t["sv_src"]):
        if cfg.pooled:
            loss = loss + model.nll_pooled_survey(t["sv_h"], t["sv_src"], t["sv_det"])
        else:
            loss = loss + model.nll_survey(t["sv_h"], t["sv_src"], t["ev_site"], t["ev_logeff"],
                                           t["ev_pos"], t["ev_neg"], t["sv_det"])
    return loss


def l2_penalty(model: OADSDM) -> torch.Tensor:
    """Squared norm of weight matrices of f and g (intercepts/observation params unpenalized)."""
    mods = [model.f] + ([model.g] if model.g is not None else [])
    return sum((p ** 2).sum() for mod in mods for n, p in mod.named_parameters() if n.endswith("weight"))


def fit(cfg: Config, batch: dict, seed: int, device: str = "cuda") -> OADSDM:
    torch.manual_seed(seed)
    np.random.seed(seed)
    t = to_tensors(batch, device)
    model = OADSDM(cfg, t["po_h"].shape[1], t["po_e"].shape[1], int(batch["n_survey"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        (objective(model, cfg, t) + cfg.l2 * l2_penalty(model)).backward()
        opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate_objective(model: OADSDM, cfg: Config, batch: dict, device: str = "cuda") -> float:
    model.eval()
    return float(objective(model, cfg, to_tensors(batch, device)))


@torch.no_grad()
def predict_eta(model: OADSDM, h: np.ndarray, device: str = "cuda") -> np.ndarray:
    model.eval()
    return model.eta(torch.as_tensor(h, dtype=torch.float32, device=device)).cpu().numpy()
