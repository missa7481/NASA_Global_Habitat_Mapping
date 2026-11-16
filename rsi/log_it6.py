#!/usr/bin/env python3
"""Record RSI iteration 6 (coherence vs accuracy) in the research memory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.log_it1_5 import COMMON  # noqa: E402
from rsi.memory import add, render  # noqa: E402

E = [
    dict(id="I6-01", iteration=6, level="Spatial-Model RSI", title="Does HiCo's hierarchy cost accuracy? Coherence vs a strong flat model",
         status="baseline-candidate", **COMMON,
         question="RF beats HiCo everywhere. Is that the price of coherence, or of HiCo's learned components?",
         hypothesis="If coherence is HiCo's value, a post-hoc HiCo normalisation of RF (or HiCo with RF as the local term) keeps RF accuracy while being coherent.",
         fusion="HiCo normalisation eta = A(P) + B(x) - logmeanexp_P B, with B = RF log-ratios; A from an independent parent-level RF (RF+Recon) "
                "or learned through the fine likelihood (HiCo-RF)",
         architecture="Arms: RF (separate fine and coarse models), RF-agg (coarse = average of fine), RF+Recon, HiCo (original), HiCo-RF; 2 seeds for neural arms",
         validation="block10, LORO, env, zone (20 splits); scored at 0.05-deg cells and 0.5-deg parents (pooled held-out counts)",
         results="Fine AUC block10/env/LORO/zone: RF 0.761/0.694/0.700/0.707; RF+Recon 0.761/0.696/0.686/0.703 (9/20 AUC wins, mean -0.004, p = 0.82); "
                 "HiCo-RF 0.768/0.683/0.661/0.653 (6/20); HiCo 0.741/0.681/0.654/0.647 (2/20, p < 0.001). "
                 "Separately fitted coarse and fine RF maps disagree: mean share TV 0.05-0.06, dominant class flips in 3.7-7.2% of parents. "
                 "Coarse accuracy of RF-agg equals the separate coarse RF (12/20, p = 0.50).",
         error_patterns="HiCo-RF is the best arm under block10 (AUC 0.768, NLL 0.910 vs RF 0.930) and worse under LORO/zone: the jointly learned macro net fits the training geography (same mechanism as I2-01).",
         improved="Coherence is free: post-hoc reconciliation removes all cross-scale contradiction at no significant accuracy cost.",
         failed="HiCo as a learner: its neural local and macro terms are the source of the deficit, not the coherence constraint.",
         causal="None; this is a prediction-level result.",
         ecological="A coarse map built as the average of the fine map is as accurate as a separately fitted coarse model, so a hierarchy only adds information when the coarse level has its own data (checklists, surveillance by admin unit).",
         open_questions="Does a coarse term fed by data the fine model cannot see (species-pool checklists) beat RF-agg at the coarse level?",
         next="Keep HiCo as the coherence operator; fill its macro term with the species-pool layer (P1), not with a learned net on parent-mean covariates."),
]

if __name__ == "__main__":
    for e in E:
        add(e)
    render()
    print(len(E), "entries")
