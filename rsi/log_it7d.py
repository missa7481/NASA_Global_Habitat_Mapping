#!/usr/bin/env python3
"""Record I7-05 (all-genus GBIF species-pool layer for HiCo-Fuse 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.log_it1_5 import COMMON  # noqa: E402
from rsi.memory import add, render  # noqa: E402

E = dict(id="I7-05", iteration=7, level="Data RSI", title="All-genus species-pool layer (GBIF 1900-2021) on the HiCo-Fuse 2 region term",
         status="baseline-candidate", **COMMON,
         question="I7-04 traced the remaining LORO error (Africa) to classes the Aedes-only pool layer does not describe. Does an all-genus pool fix it?",
         hypothesis="Adding GBIF Culicidae counts per 5-degree tile (total and Aedes/Culex/Anopheles shares, own tile and 3x3 neighbourhood; "
                    "records 1900-2021, exogenous to the 2022-2025 iNaturalist labels) lowers LORO NLL, most in Africa.",
         fusion="z(P) = Kraemer 4 + GBIF 8 features; same 28+ parameter head and trust rule (HF2-shrink2)",
         architecture="HF2+pool4 and HF2-shrink2+pool4 (head w grows to 12 x 4)",
         validation="block10, LORO, env, zone (20 splits)",
         results="Fine AUC block10/env/LORO/zone: HF2-shrink2+pool4 0.770/0.704/0.704/0.710 vs RF 0.761/0.694/0.700/0.707; "
                 "NLL 0.906/1.007/1.324/1.073 vs RF 0.930/1.040/1.289/1.113. HF2+pool4 vs RF: 14/20 AUC wins (+0.008, p = 0.12), "
                 "14/20 NLL wins (p = 0.12). LORO NLL gap to RF 0.035 (was 0.10 with the Aedes-only pool). Africa 1.958 -> 1.856, "
                 "Asia 1.360 -> 1.343, Europe 1.082 -> 1.056 (RF 1.062). GBIF data: 924 land tiles, 2.55 M records (Aedes 0.85 M, Culex 0.74 M, Anopheles 0.53 M).",
         error_patterns="Remaining LORO excess is Africa (75 test cells) and N America (1.183 vs RF 1.147).",
         improved="First HiCo variant better than RF on 14/20 splits in both ranking and log-loss while exactly coherent; the exogenous pool data, not model capacity, produced the gain.",
         failed="Not yet significant at 20 splits; LORO log-loss still slightly above RF.",
         causal="None.", ecological="Which genera have been recorded in a region (independent of the training reports) carries the regional prior that environment cannot.",
         open_questions="Finer pool resolution (country checklists, 1-degree tiles); a locked 2025 validation; Kraemer-based external checks are no longer independent for maps using this pool.",
         next="Adopt HF2-shrink2+pool4 as HiCo-Fuse 2; global maps; next: observation/detection branch and locked 2025 validation.")

if __name__ == "__main__":
    add(E)
    render()
