"""HiCo-Fuse model family.

All models output eta[x, s] = log relative intensity of target class s versus the
"other Culicidae" reference class at 0.05-degree cell x (target-group logic: the
shared observer effort cancels in the conditional class composition of reports).
Likelihood for report counts Y[x, k]:  multinomial with logits [eta_1..eta_S, 0].

Architectures
-------------
LinearEarly   eta = W [all modalities]
MLPEarly      eta = MLP([all modalities])
GatedMoE      eta_s = sum_m g_sm(context) * u_sm(enc_m(x_m))        (modality experts + gates)
HiCo          eta_s(x) = A_s(P) + log softmax_{x' in P} B_s(x') [x] + log|P|
              A: macro network on parent-mean climate (0.5 deg), B: local MLP.
              => mean over children of exp(eta) == exp(A) exactly (coherent by construction)
HiCoMoE       HiCo with B = gated modality experts over local modalities
              (Landscape, Human, AlphaEarth, Climate anomaly from parent mean).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

S = 4  # target classes; reference class has eta = 0


def mlp(d_in, d_out, hidden=64, depth=2, p=0.1):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(p)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class MoE(nn.Module):
    """Gated mixture of modality experts with per-class gates."""

    def __init__(self, dims: dict, ctx_dim: int, d=32, p=0.1):
        super().__init__()
        self.names = list(dims)
        self.enc = nn.ModuleDict({m: nn.Sequential(nn.Linear(k, d), nn.SiLU(), nn.Dropout(p), nn.Linear(d, d), nn.SiLU())
                                  for m, k in dims.items()})
        self.head = nn.ModuleDict({m: nn.Linear(d, S) for m in dims})
        self.gate = nn.Sequential(nn.Linear(ctx_dim, 32), nn.SiLU(), nn.Linear(32, S * len(dims)))
        self.bias = nn.Parameter(torch.zeros(S))

    def forward(self, xs: dict, ctx: torch.Tensor, return_parts=False):
        u = torch.stack([self.head[m](self.enc[m](xs[m])) for m in self.names], dim=1)   # N, M, S
        g = torch.softmax(self.gate(ctx).view(-1, S, len(self.names)), dim=-1).transpose(1, 2)  # N, M, S
        eta = (g * u).sum(1) + self.bias
        return (eta, g, g * u) if return_parts else eta


def parent_logsumexp(B: torch.Tensor, parent: torch.Tensor, n_par: int) -> torch.Tensor:
    """Per-parent logsumexp of B (N, S) over children; returns (n_par, S)."""
    idx = parent[:, None].expand_as(B)
    mx = torch.full((n_par, B.shape[1]), -1e30, device=B.device).scatter_reduce(0, idx, B, "amax", include_self=True)
    se = torch.zeros((n_par, B.shape[1]), device=B.device).index_add(0, parent, torch.exp(B - mx[parent]))
    return mx + torch.log(se)


class Net(nn.Module):
    def __init__(self, kind: str, mod_dims: dict, p=0.1):
        super().__init__()
        self.kind = kind
        d_all = sum(mod_dims.values())
        dc, dh = mod_dims["Climate"], mod_dims["Human"]
        if kind == "LinearEarly":
            self.f = nn.Linear(d_all, S)
        elif kind == "MLPEarly":
            self.f = mlp(d_all, S, p=p)
        elif kind.startswith("Single:"):
            self.f = mlp(mod_dims[kind.split(":")[1]], S, p=p)
        elif kind == "GatedMoE":
            self.f = MoE(mod_dims, dc + dh, p=p)
        elif kind in ("HiCo", "HiCo-climA"):
            # macro term on pooled parent representation: all modalities (default) or climate only (ablation)
            self.A = mlp(d_all if kind == "HiCo" else dc, S, hidden=32, p=p)
            self.B = mlp(d_all, S, p=p)          # local features incl. climate anomaly (same width as d_all)
        elif kind in ("HiCoMoE", "HiCoMoE-climA"):
            self.A = mlp(d_all if kind == "HiCoMoE" else dc, S, hidden=32, p=p)
            self.B = MoE({"ClimateAnomaly": dc, "Landscape": mod_dims["Landscape"], "Human": dh,
                          "AlphaEarth": mod_dims["AlphaEarth"]}, dc + dh, p=p)
        else:
            raise ValueError(kind)

    @property
    def hierarchical(self):
        return self.kind.startswith("HiCo")

    @property
    def macro_all(self):
        return self.kind in ("HiCo", "HiCoMoE")

    def flat(self, b: dict):
        if self.kind in ("LinearEarly", "MLPEarly"):
            return self.f(b["all"])
        if self.kind.startswith("Single:"):
            return self.f(b[self.kind.split(":")[1]])
        return self.f({m: b[m] for m in ("Climate", "Landscape", "Human", "AlphaEarth")}, b["ctx"])

    def local(self, b: dict, return_parts=False):
        if self.kind.startswith("HiCo") and "MoE" not in self.kind:
            x = torch.cat([b["ClimateAnomaly"], b["Landscape"], b["Human"], b["AlphaEarth"]], 1)
            return self.B(x)
        return self.B({m: b[m] for m in ("ClimateAnomaly", "Landscape", "Human", "AlphaEarth")}, b["ctx"],
                      return_parts=return_parts)

    def eta_hier(self, b_children: dict, parent: torch.Tensor, n_par: int, parent_clim: torch.Tensor, log_n: torch.Tensor):
        """eta for every child cell; parent indexes rows of parent_clim / log_n."""
        A = self.A(parent_clim)                      # n_par, S
        B = self.local(b_children)                   # N, S
        lse = parent_logsumexp(B, parent, n_par)
        return A[parent] + B - lse[parent] + log_n[parent][:, None], A


def multinomial_nll(eta: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    logits = torch.cat([eta, torch.zeros(len(eta), 1, device=eta.device)], 1)
    return -(Y * F.log_softmax(logits, 1)).sum() / Y.sum()
