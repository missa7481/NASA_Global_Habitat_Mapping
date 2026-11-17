#!/usr/bin/env python3
"""Record I7-04 (pool-space novelty trust for HiCo-Fuse 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.log_it1_5 import COMMON  # noqa: E402
from rsi.memory import add, render  # noqa: E402

E = dict(id="I7-04", iteration=7, level="Spatial-Model RSI", title="HiCo-Fuse 2 trust from species-pool novelty",
         status="baseline-candidate", **COMMON,
         question="I7-03 left Africa and part of N America overconfident. Does novelty in the species-pool layer z(P) flag them?",
         hypothesis="Same pre-declared trust rule on pool-space novelty (HF2-shrinkP) and the stricter of covariate and pool trust (HF2-shrink2) "
                    "reduce LORO NLL further than covariate trust alone.",
         fusion="As I7-01; trust per parent, exactly coherent (asserted)", architecture="HF2 head + trust = min(rho_cov, rho_pool)",
         validation="block10, LORO, env, zone (20 splits)",
         results="Fine NLL block10/env/LORO/zone: HF2-shrink2 0.911/1.017/1.349/1.088 vs RF 0.930/1.040/1.289/1.113 (HF2 1.391 LORO). "
                 "AUC 0.766/0.697/0.701/0.716 (13/20 vs RF, +0.004, p = 0.26). LORO N America NLL 1.156 (RF 1.147, HF2 1.316); "
                 "Africa unchanged 1.958 (pool trust 1.0).",
         error_patterns="Pool trust is low in N America (0.46) and Latin America (0.69) and 1.0 in Africa: Africa's Aedes pool looks familiar; "
                        "its error lies in the classes the pool layer does not cover (Culex, Anopheles).",
         improved="About 40% of the LORO log-loss gap closed; best zone and block10 log-loss of any arm; ranking gain kept.",
         failed="LORO NLL still above RF, driven by Africa (75 test cells) and Asia.",
         causal="None.", ecological="The Aedes-only pool layer cannot represent Anopheles-dominated faunas.",
         open_questions="Pool layers for Culex, Anopheles and other.",
         next="I7-05: exogenous pool layer for all classes (GBIF non-iNaturalist records), then re-run HF2-shrink2.")

if __name__ == "__main__":
    add(E)
    render()
