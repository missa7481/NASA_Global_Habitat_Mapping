"""Check OA-DSDM likelihood terms against hand-computed values."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oadsdm.model import Config, OADSDM, fit  # noqa: E402


def sig(x):
    return 1 / (1 + math.exp(-x))


class SurveyLikelihood(unittest.TestCase):
    def setUp(self):
        self.m = OADSDM(Config("t", linear=True), d_h=1, d_e=1, n_survey=1)
        with torch.no_grad():
            self.m.f.weight.zero_(); self.m.f.bias.fill_(0.3)   # eta = 0.3
            self.m.alpha.fill_(-0.1); self.m.delta.fill_(-1.0); self.m.kappa.fill_(0.5)

    def test_marginal_likelihood(self):
        # site 0: 3 events, 1 detection; site 1: 2 events, 0 detections
        h = torch.zeros(2, 1)
        src = torch.zeros(2, dtype=torch.long)
        ev_site = torch.tensor([0, 0, 0, 1, 1])
        le = torch.tensor([0.0, 1.0, -1.0, 0.0, 2.0])
        pos = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        neg = 1 - pos
        det = torch.tensor([True, False])
        got = float(self.m.nll_survey(h, src, ev_site, le, pos, neg, det))
        psi = sig(0.3 - 0.1)
        p = [sig(-1.0 + 0.5 * v) for v in le.tolist()]
        l0 = psi * p[0] * (1 - p[1]) * (1 - p[2])
        l1 = psi * (1 - p[3]) * (1 - p[4]) + (1 - psi)
        want = -(math.log(l0) + math.log(l1)) / 2
        self.assertAlmostEqual(got, want, places=5)

    def test_aggregated_binomial_equals_expanded_events(self):
        h = torch.zeros(1, 1); src = torch.zeros(1, dtype=torch.long); det = torch.tensor([True])
        agg = self.m.nll_survey(h, src, torch.tensor([0]), torch.tensor([0.4]), torch.tensor([2.0]), torch.tensor([5.0]), det)
        exp = self.m.nll_survey(h, src, torch.zeros(7, dtype=torch.long), torch.full((7,), 0.4),
                                torch.tensor([1.0, 1, 0, 0, 0, 0, 0]), torch.tensor([0.0, 0, 1, 1, 1, 1, 1]), det)
        self.assertAlmostEqual(float(agg), float(exp), places=5)

    def test_po_bias_head_absent_when_disabled(self):
        m = OADSDM(Config("t", bias_head=False), d_h=3, d_e=2, n_survey=1)
        self.assertIsNone(m.g)


class Recovery(unittest.TestCase):
    def test_linear_isdm_recovers_survey_occupancy_slope(self):
        """With strong survey data, a linear eta should recover the true slope sign/magnitude."""
        g = torch.Generator().manual_seed(0)
        n, visits = 800, 6
        h = torch.randn(n, 1, generator=g)
        z = torch.rand(n, generator=g) < torch.sigmoid(1.5 * h[:, 0])
        y = (torch.rand(n, visits, generator=g) < 0.4) & z[:, None]
        batch = dict(po_h=torch.zeros(0, 1).numpy(), po_e=torch.zeros(0, 1).numpy(), po_y=torch.zeros(0).numpy(),
                     sv_h=h.numpy(), sv_src=torch.zeros(n, dtype=torch.long).numpy(), sv_det=y.any(1).numpy(),
                     ev_site=torch.arange(n).numpy(), ev_logeff=torch.zeros(n).numpy(),
                     ev_pos=y.sum(1).float().numpy(), ev_neg=(visits - y.sum(1)).float().numpy(), n_survey=1)
        cfg = Config("rec", use_po=False, bias_head=False, linear=True, l2=0.0, epochs=600, lr=0.05)
        m = fit(cfg, batch, seed=0, device="cpu")
        self.assertAlmostEqual(float(m.f.weight), 1.5, delta=0.3)
        self.assertAlmostEqual(float(torch.sigmoid(m.delta[0])), 0.4, delta=0.05)


if __name__ == "__main__":
    unittest.main()
