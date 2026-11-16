#!/usr/bin/env python3
"""Record RSI iterations 1-5 in the research memory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.memory import add, render  # noqa: E402

COMMON = dict(data="iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features "
                   "(Climate 26, Landscape 11, Human 4, AlphaEarth 64)",
              representation="Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)")

E = [
    dict(id="I1-01", iteration=1, level="Validation RSI", title="Nine holdout designs, nine baselines", status="baseline-candidate", **COMMON,
         question="Which baselines generalise, and how optimistic is random or block CV?",
         hypothesis="Skill falls monotonically with holdout distance; spatial interpolators fall fastest.",
         fusion="Early concatenation (GLM, MLP, MLP+Coords, HGB, RF); per-class gated MoE; HiCo; kNN-geo uses coordinates only",
         validation="random5, block5/10/20, leave-one-region-out (6), environmental clusters (5), climate zones (4), temporal 2022-24 -> 2025, spatio-temporal",
         experiment="41 splits x 9 models; neural models 2 seeds",
         results="Mean AUC (random / block10 / LORO / zone / temporal): RF 0.797 / 0.766 / 0.697 / 0.703 / 0.820 (best everywhere); "
                 "MLP 0.786 / 0.757 / 0.643 / 0.698 / 0.792; HiCo 0.778 / 0.753 / 0.649 / 0.678 / 0.775; kNN-geo 0.782 / 0.733 / 0.519 / 0.657 / 0.796.",
         error_patterns="Optimism random -> LORO: RF -0.10, MLP -0.14, kNN -0.26. Europe hold-out: MLP 0.526, GatedMoE 0.538 vs GLM 0.715, RF 0.701. "
                        "Latin America weakest everywhere (0.49-0.62). Africa has 75 report cells.",
         uncertainty="LORO per-region AUC spread is large (0.49-0.76); 6 splits only.",
         improved="RF is the baseline to beat; random CV is not an acceptable headline metric.",
         failed="HiCo and GatedMoE are not better than flat models under any transfer design.",
         next="Decompose transfer error (I1-02, I1-03)."),
    dict(id="I1-02", iteration=1, level="Spatial-Model RSI", title="Error vs distance, density and environmental novelty", status="baseline-candidate", **COMMON,
         question="Are errors driven by geographic distance, sampling density or environmental novelty?",
         experiment="Per-cell NLL of out-of-fold predictions vs support context; joint weighted OLS of the (model - kNN) NLL gap",
         results="NLL rises with distance for all models; beyond 1,000 km (LORO) MLP 1.21 and RF 1.15 vs class prior 1.17. "
                 "The covariate-model advantage over kNN grows with log distance (b = -0.12 to -0.20 under LORO); novelty adds a smaller term.",
         error_patterns="Within 10 km, kNN is competitive; covariate models dominate from 200 km out.",
         ecological="Nearby reports carry the local species pool; distant transfer must rely on niche covariates.",
         next="Ask why models barely beat the prior in NLL far from data (I1-03)."),
    dict(id="I1-03", iteration=1, level="Scientific RSI", title="Transfer loss is class-prior (species-pool) shift", status="baseline-candidate", **COMMON,
         question="Is the transfer failure a changed habitat relation p(y|x) or a shifted regional class prior p(y)?",
         experiment="Re-weight the same OOF predictions with (a) true held-out class frequencies (oracle), (b) Saerens EM label-shift estimate",
         results="LORO NLL, RF: 1.302 -> 1.070 with the oracle prior; held-out prior alone 1.057. HGB 1.376 -> 1.133; MLP 1.428 -> 1.239. Block10: no change.",
         failed="EM label-shift correction diverged (NLL 3-13): with miscalibrated conditionals it pushes class priors to ~0.",
         ecological="Which species have arrived in a region (introduction history, regional pool) dominates composition across regions; it is not a niche quantity.",
         causal="Consistent with an unmeasured spatial confounder I (introduction history) in the DAG; region-level priors are not identifiable from environment.",
         open_questions="Can an exogenous pool signal (checklists, literature compendia) supply the prior?",
         next="I5-01 exogenous species-pool covariate."),
    dict(id="I2-01", iteration=2, level="Fusion RSI", title="Seven weighting strategies for five experts", status="rejected", **COMMON,
         question="Should information sources be weighted unequally, and should weights vary in space?",
         fusion="Log-linear pool of modality experts (Climate, Landscape, Human, AlphaEarth MLPs) + kNN-geo",
         weighting="Equal; theory-fixed; stacked global; stacked by region (shrunk); attention on env context; attention on env + support context; inverse seed variance; local reliability",
         validation="block10, LORO, env clusters; learned weights fitted on inner 10-degree-block OOF of the training set",
         results="Inner-OOF NLL gain over Equal: +0.035 to +0.106 for learned schemes. Outer gain under LORO: Stack-global -0.059, Stack-region -0.023, "
                 "Attn-env -0.168, Attn-support -0.197, Reliability +0.039 (only positive). LORO AUC: Theory 0.666 (best), Equal 0.636, Stack 0.608.",
         error_patterns="Stacking learns w_Geo 0.42-0.49 because inner OOF is an interpolation regime; the support gate assigned Geo MORE weight at 50-1,000 km (0.73-0.77) under LORO.",
         failed="All free learned weightings overfit the training geography; inverse-variance weighting poor (seed variance is not comparable across experts).",
         improved="Identified the mechanism: weights are learned in a different support regime from the one they are deployed in.",
         next="Regime-matched inner OOF and monotone support constraints (I4-01)."),
    dict(id="I3-01", iteration=3, level="Scientific RSI", title="Region-to-region transfer matrix by modality", status="inconclusive", **COMMON,
         question="Which modality's relation to composition is stable across regions (candidate driver) vs region-specific (proxy)?",
         experiment="Single-modality and all-feature MLPs trained in one region, evaluated in the others",
         results="N America -> Europe AUC: Climate 0.659, All 0.592, AlphaEarth 0.571, Human 0.526. Europe -> N America: Climate 0.660, AE 0.642. "
                 "Human transfers near chance from every region. Oceania-trained models fail everywhere (0.37-0.60).",
         causal="Climate relations transfer between the two temperate invasion fronts better than proxies, consistent with a physiological C -> Z effect; not identification.",
         uncertainty="Single seed pair per cell of the matrix; small regions noisy.",
         next="Invariance training and proxy decomposition (I3-02, I3-03b)."),
    dict(id="I3-02", iteration=3, level="Spatial-Model RSI", title="Invariance penalties and causal feature subsets", status="rejected", **COMMON,
         question="Do invariance objectives or causally motivated feature subsets improve transfer?",
         architecture="MLP with V-REx (lambda 1, 10) or soft group-DRO over training regions; subsets Drivers (Clim+Land), +Human, +AE",
         validation="block10, LORO, env, zone",
         results="LORO AUC: ERM 0.687, V-REx 0.684, GroupDRO 0.674, Drivers only 0.664, Drivers+Human 0.656. Zone: V-REx 0.685 vs ERM 0.683.",
         failed="Invariance penalties do not improve transfer; dropping AlphaEarth hurts every design.",
         causal="Prior shift (I1-03), not spurious features, dominates the transfer error, which is why invariance over x does not help.",
         next="Treat region-level prior explicitly rather than penalising risk variance."),
    dict(id="I3-03", iteration=3, level="Representation RSI", title="AlphaEarth proxy decomposition (corrected)", status="baseline-candidate", **COMMON,
         question="Is AlphaEarth's value a re-encoding of named drivers or residual surface structure?",
         experiment="Each component alone: AE, AE_hat (ridge on Clim+Land+Human), AE_perp (residual)",
         results="Drivers explain 11% of AE variance (linear). AUC block10 / LORO / zone: AE 0.751 / 0.624 / 0.685; AE_hat 0.691 / 0.603 / 0.622; "
                 "AE_perp 0.625 / 0.534 / 0.555; Drivers+Human 0.713 / 0.656 / 0.658.",
         failed="The first version (I3-03a, Drivers+AE vs Drivers+AE_perp) was a linear reparameterisation and could not discriminate; superseded.",
         ecological="AE's residual carries real in-region signal that does not transfer: local surface structure, or region identity (introduction history) leaking through.",
         causal="AE acts as a high-capacity region fingerprint; it is not evidence for a transferable habitat mechanism.",
         next="Let AE dominate near data and fade under novelty (I4-01)."),
    dict(id="I4-01", iteration=4, level="Spatial-Model RSI", title="Support-decay fusion with regime-matched training", status="inconclusive", **COMMON,
         question="Do monotone support constraints and regime-matched inner OOF fix the weighting failure of I2-01?",
         hypothesis="H4a: weights learned on inner OOF that includes leave-training-region-out transfer better. H4b: monotone decay "
                    "(Geo weight exp(-d/rho); covariate experts fade with env novelty towards the prior) beats a free gate.",
         architecture="Log-linear pool of 6 experts (4 modality MLPs, all-feature MLP, kNN-geo) with ~15 parameters",
         validation="block10, LORO, env, zone; 20 splits; paired sign tests",
         results="Decay[mixed] vs free support gate: better in 17/20 splits on AUC and NLL (p = 0.003). vs Equal: 10/20 (p = 1.0). vs RF: 8/20. "
                 "Learned correlation range rho: 7,600-22,000 km with block inner OOF, ~700 km with mixed; LORO Geo weight 0.32 -> 0.11, prior share 0.12 -> 0.56.",
         improved="Monotone structure is reliably better than free gating; rho becomes ecologically interpretable when trained on the right regime.",
         failed="No significant gain over equal pooling or RF; regime matching changes parameters but not accuracy (12/20).",
         uncertainty="LORO per-split differences range -0.04 to +0.13 AUC; 6 regions are too few to resolve gains < 0.02.",
         next="Fusion weighting is not the binding constraint; the regional prior is (I1-03)."),
    dict(id="I5-01", iteration=5, level="Data RSI", title="Exogenous species-pool covariate (Kraemer)", status="inconclusive", data=COMMON["data"],
         question="Does an independent coarse species-pool signal restore transfer?",
         hypothesis="Gain under LORO/zone > gain under block10. Falsified if not.",
         representation="Base + log1p Kraemer compendium cells within 100 / 500 km for Ae. aegypti and Ae. albopictus",
         validation="block10, LORO, zone, env",
         results="MLP AUC gain: block10 +0.015, LORO +0.041, zone +0.019. RF: +0.007 / +0.010 / +0.002. LORO NLL barely changes (MLP 1.357 -> 1.344; RF 1.289 -> 1.314).",
         improved="Prediction holds for the MLP: transfer gain about 3x the in-distribution gain.",
         failed="Calibration under transfer not fixed: the pool signal covers only 2 of 5 classes.",
         uncertainty="MLP Europe hold-out AUC 0.526 (I1) vs 0.705 here with only seeds changed: neural extrapolation is seed-unstable.",
         open_questions="Can pool information for Culex, Anopheles and 'other' be obtained (country checklists, GBIF non-iNat datasets)?",
         next="Data RSI: species-pool layer for all classes; Spatial-Model RSI: pool term as the macro level of HiCo (see proposal P1)."),
]

if __name__ == "__main__":
    for e in E:
        add(e)
    render()
    print(len(E), "entries")
