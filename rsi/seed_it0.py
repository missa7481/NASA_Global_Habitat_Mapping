#!/usr/bin/env python3
"""Iteration 0: record the evidence accumulated before the RSI loop (OA-DSDM and HiCo-Fuse work)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.memory import add, render  # noqa: E402

E = [
    dict(id="I0-01", iteration=0, level="Representation RSI", title="AlphaEarth leaks human footprint", status="baseline-candidate",
         question="Does the AlphaEarth embedding encode observer-effort covariates?",
         hypothesis="Satellite embeddings re-encode settlement, so effort bias can enter through them.",
         data="2,462 iNat/NEON/Austria sites; AlphaEarth 64-d 2024; WorldPop, built, VIIRS, accessibility",
         experiment="Ridge/MLP regression AE -> effort covariates; PB model with/without effort subspace",
         results="AE -> built R2 0.876, population 0.821, night lights 0.816. PB AP 0.781 -> 0.625 when the effort subspace is removed.",
         causal="AE is a descendant of Human (and of Climate/Landscape); conditioning on it opens observation-bias paths.",
         improved="Established that 'effort-free' representations cannot be assumed.",
         open_questions="How much of AE's predictive value is proxy vs residual surface structure?",
         next="Proxy decomposition of AE (I3-03)."),
    dict(id="I0-02", iteration=0, level="Spatial-Model RSI", title="Observation-aware ISDM (OA-DSDM)", status="inconclusive",
         question="Does jointly modelling presence-only and survey data with a reporting-bias head improve transfer?",
         hypothesis="Surveys spanning the effort gradient identify the bias head g(e), removing effort confounding.",
         architecture="Thinned point process + occupancy-detection likelihood + bias head; variants orth/multiscale",
         validation="Pre-registered: NEON 2026 held-out clusters (AUC/AP), siteID cluster bootstrap",
         results="Untuned OA-DSDM NEON AUC 0.874 vs PB 0.883 (GLM 0.899). Tuned: AP ~0.38-0.40 vs PB 0.31 but all CIs span 0. Multiscale AE: NEON AUC 0.907 (best). Semi-synthetic (gamma=2): PB rho 0.40, OA-DSDM 0.67, linear 0.80.",
         failed="No significant real-data gain; Austria survey only weakly learnable (AUC 0.65-0.69).",
         uncertainty="Survey coverage of the urban gradient is the binding constraint (NEON sites have pop = built = 0).",
         next="AIMSurv 2020 (1,405 traps, 24 countries) as a survey spanning the urban gradient - pending covariate extraction (lst_night drift)."),
    dict(id="I0-03", iteration=0, level="Fusion RSI", title="HiCo-Fuse fusion comparison (10-degree block CV)", status="baseline-candidate",
         question="Which fusion of Climate / Landscape / Human / AlphaEarth predicts reported mosquito composition best?",
         data="iNat research-grade Culicidae 2022-2025, <=2 km; 7,004 report cells; 105 features at 0.05 degrees",
         fusion="Early concat (linear, MLP); per-class gated MoE; late rank ensemble; single modalities",
         validation="10-degree spatial blocks, 5 folds",
         results="Mean AUC: MLPEarly 0.735, AE-only 0.732, Linear 0.724, GatedMoE 0.722, LateRank 0.704; Climate 0.618, Landscape 0.646, Human 0.614.",
         error_patterns="Culex and Anopheles near chance (0.55 / 0.64): genus-level classes mix ecologically different species.",
         failed="Gated MoE and late rank blending did not beat early concatenation.",
         open_questions="Is the MLP's edge interpolation (block CV) rather than transfer?",
         next="Multi-design holdouts (I1-01)."),
    dict(id="I0-04", iteration=0, level="Spatial-Model RSI", title="Hierarchically coherent map (HiCo)", status="baseline-candidate",
         question="Can a fine map be exactly coherent with its coarse level?",
         architecture="eta_s(x) = A_s(parent) + B_s(x) - logsumexp_P B_s + log|P|; log-mean-exp ensembling",
         results="Coherence error 0 exactly; a separately fitted 0.5-degree model is off by >2x in 66-88% of regions. Cost: CV AUC 0.722 vs 0.735 flat; two-part external AUC Kraemer aegypti 0.849, albopictus 0.869, NEON 0.921.",
         improved="Guaranteed global-to-local consistency (the user's validity requirement).",
         failed="Small accuracy cost at fine scale; climate-only macro term 0.687.",
         next="Keep as the map-construction layer; improve the local term instead."),
    dict(id="I0-05", iteration=0, level="Data RSI", title="Kraemer as multi-resolution evidence", status="rejected",
         question="Do Kraemer compendium records help when used as fine or coarse evidence?",
         results="As fine points: albopictus 0.825 -> 0.783 (tropics 0.675 -> 0.594). As coarse evidence on A: neutral.",
         failed="Literature records 1960-2014 add sampling-era bias; kept only for external validation.",
         next="Do not reuse as training data without a temporal/sampling model."),
    dict(id="I0-06", iteration=0, level="Representation RSI", title="Vision-language concepts (RemoteCLIP)", status="inconclusive",
         question="Do zero-shot habitat concepts from Sentinel-2 chips add information beyond AlphaEarth?",
         results="Concepts alone AUC 0.70 (albopictus) / 0.80 (aegypti); +0.002 at most over AE at 0.05 degrees. Distillation R2: land cover 0.5-0.65, breeding-site concepts 0.19-0.29.",
         ecological="aegypti cells enriched for dense urban (+0.96 SD), informal settlement, tyre dump imagery.",
         failed="No predictive gain at 5 km support.",
         next="Re-test at sub-km (trap) support with finer imagery."),
    dict(id="I0-07", iteration=0, level="Spatial-Model RSI", title="Two-part map: support x composition", status="baseline-candidate",
         question="Why did composition-only scores fail externally (Kraemer AUC ~0.5)?",
         hypothesis="Composition extrapolates into areas with no mosquito community (Sahara).",
         architecture="S_s = sigmoid(h(x)) * pi_s(x); h = community support with effort held at its mean",
         results="Kraemer AUC 0.85-0.87 after adding support. Support block-CV AUC 0.965 (effort) / 0.906 (effort at mean).",
         causal="Holding effort at its mean is a do()-style intervention only if the effort term is not also habitat (it partly is).",
         next="Keep; the support term is the place to test effort-invariance."),
]

if __name__ == "__main__":
    for e in E:
        add(e)
    render()
    print(len(E), "entries")
