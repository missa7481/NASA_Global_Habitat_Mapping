#!/usr/bin/env python3
"""Record RSI iteration 7 (HiCo-Fuse 2 rebuild) in the research memory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.log_it1_5 import COMMON  # noqa: E402
from rsi.memory import add, render  # noqa: E402

E = [
    dict(id="I7-01", iteration=7, level="Spatial-Model RSI", title="HiCo-Fuse 2: coherent hierarchy over an RF niche term with a species-pool macro term",
         status="baseline-candidate", **COMMON,
         question="After I6-01 (coherence is free; HiCo's neural terms lose), can a rebuilt HiCo-Fuse match or beat RF while staying exactly coherent?",
         hypothesis="Replacing HiCo's learned terms with a fixed RF local term and a 28-parameter macro head (alpha, gamma, tau, pool weights) "
                    "fitted on spatially out-of-fold RF output keeps RF accuracy and adds transfer gain through the exogenous pool layer. "
                    "Falsified if HF2 loses to RF under LORO.",
         fusion="eta_s = A_s(P) + tau_s B_s(x) - logmeanexp_P(tau_s B_s); A_s = alpha_s + gamma_s M_s(P) + w_s.z(P); "
                "B = RF log-ratios (all four modalities); M = RF-aggregated parent map; z = Kraemer pool density at the parent (100/500 km, 2 species)",
         architecture="Arms: RF, RF+Recon (I6), HF2-cal (no pool), HF2; head = ridge (lambda 0.01) towards identity, LBFGS, "
                      "fitted on 4-fold 5-degree-block OOF RF inside the training set; coherence asserted < 1e-6 on every split",
         validation="block10, LORO, env, zone (20 splits); 0.05-deg cells and 0.5-deg parents",
         results="Fine AUC block10/env/LORO/zone: HF2 0.766/0.701/0.702/0.710 vs RF 0.761/0.694/0.700/0.707 (best on all four designs; "
                 "13/20 wins, mean +0.004, p = 0.26). Fine NLL: HF2 0.912/1.022/1.391/1.096 vs RF 0.930/1.040/1.289/1.113 "
                 "(better except LORO; 11/20). Largest LORO AUC gains: Latin America +0.046, Europe +0.017, Oceania +0.015; "
                 "losses Africa -0.055 (75 test cells), N America -0.013. HF2-cal (no pool) = RF (9/20). Coarse AUC HF2 vs RF-agg: "
                 "LORO 0.672 vs 0.654, zone 0.684 vs 0.679.",
         error_patterns="Head sharpens the Aedes macro term (gamma 1.2-1.5 for both Aedes) and flattens Culex (gamma ~0.8): ranking improves, "
                        "but held-out-region probabilities become overconfident (LORO NLL +0.10).",
         improved="First HiCo variant that is not worse than RF on any design by AUC while exactly coherent; the pool term, not calibration, carries the gain.",
         failed="Calibration under region hold-out (LORO NLL worse in 4/6 regions).",
         causal="None; prediction-level.",
         ecological="The exogenous Aedes pool density helps most where Aedes arrival history differs from the training regions (Latin America, Europe).",
         open_questions="How to keep the pool ranking gain without overconfidence: shrink gamma under detected novelty, or pool data for Culex/Anopheles/other.",
         next="I7-02 tested larger OOF blocks; next: novelty-dependent shrinkage of the head towards identity, and pool layers for the other classes."),
    dict(id="I7-02", iteration=7, level="Spatial-Model RSI", title="HiCo-Fuse 2 head fitted on 20-degree OOF blocks",
         status="rejected", **COMMON,
         question="Is HF2's LORO overconfidence caused by OOF folds (5-degree blocks) being less shifted than deployment?",
         hypothesis="Fitting the head on 20-degree-block OOF lowers gamma and fixes LORO NLL without losing AUC.",
         fusion="As I7-01", architecture="As I7-01 with 20-degree OOF blocks",
         validation="block10, LORO, env, zone (20 splits)",
         results="LORO NLL worse, not better: 1.490 (vs 1.391 with 5-degree blocks, RF 1.289); zone NLL 1.142 (vs 1.096). AUC about unchanged (12/20 vs RF).",
         improved="Nothing.", failed="Hypothesis falsified: coarser OOF folds give fewer, noisier folds and a worse head.",
         causal="None.", ecological="None.",
         open_questions="The overconfidence is not an OOF-shift-mismatch artefact; it is regional.",
         next="Keep 5-degree OOF. Do not retry block-size changes without new evidence."),
]

if __name__ == "__main__":
    for e in E:
        add(e)
    render()
    print(len(E), "entries")
