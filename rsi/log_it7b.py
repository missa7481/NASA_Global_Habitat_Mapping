#!/usr/bin/env python3
"""Record I7-03 (novelty-dependent shrinkage of the HiCo-Fuse 2 head)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.log_it1_5 import COMMON  # noqa: E402
from rsi.memory import add, render  # noqa: E402

E = dict(id="I7-03", iteration=7, level="Spatial-Model RSI", title="HiCo-Fuse 2 with novelty-dependent shrinkage towards RF",
         status="inconclusive", **COMMON,
         question="Does pulling the HF2 head back to plain RF in environmentally novel parents fix its LORO overconfidence (I7-01)?",
         hypothesis="Head trust rho(P) = clip(2 - d(P)/q90, 0, 1), d = mean distance of the parent's mean features to its 5 nearest training parents, "
                    "q90 = 90th percentile of OOF novelty seen while fitting the head (rule fixed before the run). Predicts LORO NLL <= RF with AUC kept.",
         fusion="As I7-01; all head parameters scaled per parent by rho, so the map stays exactly coherent (asserted < 1e-6)",
         architecture="HF2-shrink = HF2 head + per-parent trust; no new fitted parameters",
         validation="block10, LORO, env, zone (20 splits)",
         results="Fine NLL block10/env/LORO/zone: 0.912/1.019/1.366/1.091 (HF2 0.912/1.022/1.391/1.096; RF 0.930/1.040/1.289/1.113). "
                 "AUC 0.765/0.699/0.699/0.711 (12/20 vs RF). Mean trust: block10 0.96-0.99, LORO 0.70-0.95. "
                 "LORO NLL by region (RF / HF2 / shrink): N America 1.147/1.316/1.231, Asia 1.293/1.445/1.384, Africa 1.691/1.958/1.958, Europe 1.062/1.082/1.082.",
         error_patterns="The rule engages where environmental novelty is high (N America, Asia) and recovers about half of their loss, "
                        "but Africa (trust 0.88) keeps the full loss: its failure is species-pool novelty, which covariate distance cannot see.",
         improved="Closes about a quarter of the LORO log-loss gap at no in-distribution cost.",
         failed="Does not reach RF calibration under LORO.",
         causal="None.", ecological="Consistent with I1-03: the residual error is a regional class-prior shift, not an environmental one.",
         open_questions="Can trust be driven by pool novelty (distance in z(P), or Kraemer/checklist coverage) rather than covariate novelty?",
         next="Shrink on pool-space novelty; pool layers for Culex/Anopheles/other (P1).")

if __name__ == "__main__":
    add(E)
    render()
