# RSI research memory

Generated from `ledger.jsonl`; do not edit by hand.

| id | iteration | level | title | status |
|---|---|---|---|---|
| I0-01 | 0 | Representation RSI | AlphaEarth leaks human footprint | baseline-candidate |
| I0-02 | 0 | Spatial-Model RSI | Observation-aware ISDM (OA-DSDM) | inconclusive |
| I0-03 | 0 | Fusion RSI | HiCo-Fuse fusion comparison (10-degree block CV) | baseline-candidate |
| I0-04 | 0 | Spatial-Model RSI | Hierarchically coherent map (HiCo) | baseline-candidate |
| I0-05 | 0 | Data RSI | Kraemer as multi-resolution evidence | rejected |
| I0-06 | 0 | Representation RSI | Vision-language concepts (RemoteCLIP) | inconclusive |
| I0-07 | 0 | Spatial-Model RSI | Two-part map: support x composition | baseline-candidate |
| I1-01 | 1 | Validation RSI | Nine holdout designs, nine baselines | baseline-candidate |
| I1-02 | 1 | Spatial-Model RSI | Error vs distance, density and environmental novelty | baseline-candidate |
| I1-03 | 1 | Scientific RSI | Transfer loss is class-prior (species-pool) shift | baseline-candidate |
| I2-01 | 2 | Fusion RSI | Seven weighting strategies for five experts | rejected |
| I3-01 | 3 | Scientific RSI | Region-to-region transfer matrix by modality | inconclusive |
| I3-02 | 3 | Spatial-Model RSI | Invariance penalties and causal feature subsets | rejected |
| I3-03 | 3 | Representation RSI | AlphaEarth proxy decomposition (corrected) | baseline-candidate |
| I4-01 | 4 | Spatial-Model RSI | Support-decay fusion with regime-matched training | inconclusive |
| I5-01 | 5 | Data RSI | Exogenous species-pool covariate (Kraemer) | inconclusive |
| P1 | 6 | Spatial-Model RSI | Proposal: Pool-Niche-Support (PNS) hierarchical model | proposed |
| M1 | 6 | Scientific RSI | Meta-lessons for the research strategy | infrastructure |
| I6-01 | 6 | Spatial-Model RSI | Does HiCo's hierarchy cost accuracy? Coherence vs a strong flat model | baseline-candidate |

## I0-01 - AlphaEarth leaks human footprint

*Iteration 0 - Representation RSI - status: **baseline-candidate***

- **question:** Does the AlphaEarth embedding encode observer-effort covariates?
- **hypothesis:** Satellite embeddings re-encode settlement, so effort bias can enter through them.
- **data:** 2,462 iNat/NEON/Austria sites; AlphaEarth 64-d 2024; WorldPop, built, VIIRS, accessibility
- **experiment:** Ridge/MLP regression AE -> effort covariates; PB model with/without effort subspace
- **results:** AE -> built R2 0.876, population 0.821, night lights 0.816. PB AP 0.781 -> 0.625 when the effort subspace is removed.
- **causal:** AE is a descendant of Human (and of Climate/Landscape); conditioning on it opens observation-bias paths.
- **improved:** Established that 'effort-free' representations cannot be assumed.
- **open questions:** How much of AE's predictive value is proxy vs residual surface structure?
- **next:** Proxy decomposition of AE (I3-03).

## I0-02 - Observation-aware ISDM (OA-DSDM)

*Iteration 0 - Spatial-Model RSI - status: **inconclusive***

- **question:** Does jointly modelling presence-only and survey data with a reporting-bias head improve transfer?
- **hypothesis:** Surveys spanning the effort gradient identify the bias head g(e), removing effort confounding.
- **architecture:** Thinned point process + occupancy-detection likelihood + bias head; variants orth/multiscale
- **validation:** Pre-registered: NEON 2026 held-out clusters (AUC/AP), siteID cluster bootstrap
- **results:** Untuned OA-DSDM NEON AUC 0.874 vs PB 0.883 (GLM 0.899). Tuned: AP ~0.38-0.40 vs PB 0.31 but all CIs span 0. Multiscale AE: NEON AUC 0.907 (best). Semi-synthetic (gamma=2): PB rho 0.40, OA-DSDM 0.67, linear 0.80.
- **uncertainty:** Survey coverage of the urban gradient is the binding constraint (NEON sites have pop = built = 0).
- **failed:** No significant real-data gain; Austria survey only weakly learnable (AUC 0.65-0.69).
- **next:** AIMSurv 2020 (1,405 traps, 24 countries) as a survey spanning the urban gradient - pending covariate extraction (lst_night drift).

## I0-03 - HiCo-Fuse fusion comparison (10-degree block CV)

*Iteration 0 - Fusion RSI - status: **baseline-candidate***

- **question:** Which fusion of Climate / Landscape / Human / AlphaEarth predicts reported mosquito composition best?
- **data:** iNat research-grade Culicidae 2022-2025, <=2 km; 7,004 report cells; 105 features at 0.05 degrees
- **fusion:** Early concat (linear, MLP); per-class gated MoE; late rank ensemble; single modalities
- **validation:** 10-degree spatial blocks, 5 folds
- **results:** Mean AUC: MLPEarly 0.735, AE-only 0.732, Linear 0.724, GatedMoE 0.722, LateRank 0.704; Climate 0.618, Landscape 0.646, Human 0.614.
- **error patterns:** Culex and Anopheles near chance (0.55 / 0.64): genus-level classes mix ecologically different species.
- **failed:** Gated MoE and late rank blending did not beat early concatenation.
- **open questions:** Is the MLP's edge interpolation (block CV) rather than transfer?
- **next:** Multi-design holdouts (I1-01).

## I0-04 - Hierarchically coherent map (HiCo)

*Iteration 0 - Spatial-Model RSI - status: **baseline-candidate***

- **question:** Can a fine map be exactly coherent with its coarse level?
- **architecture:** eta_s(x) = A_s(parent) + B_s(x) - logsumexp_P B_s + log|P|; log-mean-exp ensembling
- **results:** Coherence error 0 exactly; a separately fitted 0.5-degree model is off by >2x in 66-88% of regions. Cost: CV AUC 0.722 vs 0.735 flat; two-part external AUC Kraemer aegypti 0.849, albopictus 0.869, NEON 0.921.
- **improved:** Guaranteed global-to-local consistency (the user's validity requirement).
- **failed:** Small accuracy cost at fine scale; climate-only macro term 0.687.
- **next:** Keep as the map-construction layer; improve the local term instead.

## I0-05 - Kraemer as multi-resolution evidence

*Iteration 0 - Data RSI - status: **rejected***

- **question:** Do Kraemer compendium records help when used as fine or coarse evidence?
- **results:** As fine points: albopictus 0.825 -> 0.783 (tropics 0.675 -> 0.594). As coarse evidence on A: neutral.
- **failed:** Literature records 1960-2014 add sampling-era bias; kept only for external validation.
- **next:** Do not reuse as training data without a temporal/sampling model.

## I0-06 - Vision-language concepts (RemoteCLIP)

*Iteration 0 - Representation RSI - status: **inconclusive***

- **question:** Do zero-shot habitat concepts from Sentinel-2 chips add information beyond AlphaEarth?
- **results:** Concepts alone AUC 0.70 (albopictus) / 0.80 (aegypti); +0.002 at most over AE at 0.05 degrees. Distillation R2: land cover 0.5-0.65, breeding-site concepts 0.19-0.29.
- **ecological:** aegypti cells enriched for dense urban (+0.96 SD), informal settlement, tyre dump imagery.
- **failed:** No predictive gain at 5 km support.
- **next:** Re-test at sub-km (trap) support with finer imagery.

## I0-07 - Two-part map: support x composition

*Iteration 0 - Spatial-Model RSI - status: **baseline-candidate***

- **question:** Why did composition-only scores fail externally (Kraemer AUC ~0.5)?
- **hypothesis:** Composition extrapolates into areas with no mosquito community (Sahara).
- **architecture:** S_s = sigmoid(h(x)) * pi_s(x); h = community support with effort held at its mean
- **results:** Kraemer AUC 0.85-0.87 after adding support. Support block-CV AUC 0.965 (effort) / 0.906 (effort at mean).
- **causal:** Holding effort at its mean is a do()-style intervention only if the effort term is not also habitat (it partly is).
- **next:** Keep; the support term is the place to test effort-invariance.

## I1-01 - Nine holdout designs, nine baselines

*Iteration 1 - Validation RSI - status: **baseline-candidate***

- **question:** Which baselines generalise, and how optimistic is random or block CV?
- **hypothesis:** Skill falls monotonically with holdout distance; spatial interpolators fall fastest.
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)
- **fusion:** Early concatenation (GLM, MLP, MLP+Coords, HGB, RF); per-class gated MoE; HiCo; kNN-geo uses coordinates only
- **validation:** random5, block5/10/20, leave-one-region-out (6), environmental clusters (5), climate zones (4), temporal 2022-24 -> 2025, spatio-temporal
- **experiment:** 41 splits x 9 models; neural models 2 seeds
- **results:** Mean AUC (random / block10 / LORO / zone / temporal): RF 0.797 / 0.766 / 0.697 / 0.703 / 0.820 (best everywhere); MLP 0.786 / 0.757 / 0.643 / 0.698 / 0.792; HiCo 0.778 / 0.753 / 0.649 / 0.678 / 0.775; kNN-geo 0.782 / 0.733 / 0.519 / 0.657 / 0.796.
- **error patterns:** Optimism random -> LORO: RF -0.10, MLP -0.14, kNN -0.26. Europe hold-out: MLP 0.526, GatedMoE 0.538 vs GLM 0.715, RF 0.701. Latin America weakest everywhere (0.49-0.62). Africa has 75 report cells.
- **uncertainty:** LORO per-region AUC spread is large (0.49-0.76); 6 splits only.
- **improved:** RF is the baseline to beat; random CV is not an acceptable headline metric.
- **failed:** HiCo and GatedMoE are not better than flat models under any transfer design.
- **next:** Decompose transfer error (I1-02, I1-03).

## I1-02 - Error vs distance, density and environmental novelty

*Iteration 1 - Spatial-Model RSI - status: **baseline-candidate***

- **question:** Are errors driven by geographic distance, sampling density or environmental novelty?
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)
- **experiment:** Per-cell NLL of out-of-fold predictions vs support context; joint weighted OLS of the (model - kNN) NLL gap
- **results:** NLL rises with distance for all models; beyond 1,000 km (LORO) MLP 1.21 and RF 1.15 vs class prior 1.17. The covariate-model advantage over kNN grows with log distance (b = -0.12 to -0.20 under LORO); novelty adds a smaller term.
- **error patterns:** Within 10 km, kNN is competitive; covariate models dominate from 200 km out.
- **ecological:** Nearby reports carry the local species pool; distant transfer must rely on niche covariates.
- **next:** Ask why models barely beat the prior in NLL far from data (I1-03).

## I1-03 - Transfer loss is class-prior (species-pool) shift

*Iteration 1 - Scientific RSI - status: **baseline-candidate***

- **question:** Is the transfer failure a changed habitat relation p(y|x) or a shifted regional class prior p(y)?
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)
- **experiment:** Re-weight the same OOF predictions with (a) true held-out class frequencies (oracle), (b) Saerens EM label-shift estimate
- **results:** LORO NLL, RF: 1.302 -> 1.070 with the oracle prior; held-out prior alone 1.057. HGB 1.376 -> 1.133; MLP 1.428 -> 1.239. Block10: no change.
- **ecological:** Which species have arrived in a region (introduction history, regional pool) dominates composition across regions; it is not a niche quantity.
- **causal:** Consistent with an unmeasured spatial confounder I (introduction history) in the DAG; region-level priors are not identifiable from environment.
- **failed:** EM label-shift correction diverged (NLL 3-13): with miscalibrated conditionals it pushes class priors to ~0.
- **open questions:** Can an exogenous pool signal (checklists, literature compendia) supply the prior?
- **next:** I5-01 exogenous species-pool covariate.

## I2-01 - Seven weighting strategies for five experts

*Iteration 2 - Fusion RSI - status: **rejected***

- **question:** Should information sources be weighted unequally, and should weights vary in space?
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)
- **fusion:** Log-linear pool of modality experts (Climate, Landscape, Human, AlphaEarth MLPs) + kNN-geo
- **weighting:** Equal; theory-fixed; stacked global; stacked by region (shrunk); attention on env context; attention on env + support context; inverse seed variance; local reliability
- **validation:** block10, LORO, env clusters; learned weights fitted on inner 10-degree-block OOF of the training set
- **results:** Inner-OOF NLL gain over Equal: +0.035 to +0.106 for learned schemes. Outer gain under LORO: Stack-global -0.059, Stack-region -0.023, Attn-env -0.168, Attn-support -0.197, Reliability +0.039 (only positive). LORO AUC: Theory 0.666 (best), Equal 0.636, Stack 0.608.
- **error patterns:** Stacking learns w_Geo 0.42-0.49 because inner OOF is an interpolation regime; the support gate assigned Geo MORE weight at 50-1,000 km (0.73-0.77) under LORO.
- **improved:** Identified the mechanism: weights are learned in a different support regime from the one they are deployed in.
- **failed:** All free learned weightings overfit the training geography; inverse-variance weighting poor (seed variance is not comparable across experts).
- **next:** Regime-matched inner OOF and monotone support constraints (I4-01).

## I3-01 - Region-to-region transfer matrix by modality

*Iteration 3 - Scientific RSI - status: **inconclusive***

- **question:** Which modality's relation to composition is stable across regions (candidate driver) vs region-specific (proxy)?
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)
- **experiment:** Single-modality and all-feature MLPs trained in one region, evaluated in the others
- **results:** N America -> Europe AUC: Climate 0.659, All 0.592, AlphaEarth 0.571, Human 0.526. Europe -> N America: Climate 0.660, AE 0.642. Human transfers near chance from every region. Oceania-trained models fail everywhere (0.37-0.60).
- **uncertainty:** Single seed pair per cell of the matrix; small regions noisy.
- **causal:** Climate relations transfer between the two temperate invasion fronts better than proxies, consistent with a physiological C -> Z effect; not identification.
- **next:** Invariance training and proxy decomposition (I3-02, I3-03b).

## I3-02 - Invariance penalties and causal feature subsets

*Iteration 3 - Spatial-Model RSI - status: **rejected***

- **question:** Do invariance objectives or causally motivated feature subsets improve transfer?
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)
- **architecture:** MLP with V-REx (lambda 1, 10) or soft group-DRO over training regions; subsets Drivers (Clim+Land), +Human, +AE
- **validation:** block10, LORO, env, zone
- **results:** LORO AUC: ERM 0.687, V-REx 0.684, GroupDRO 0.674, Drivers only 0.664, Drivers+Human 0.656. Zone: V-REx 0.685 vs ERM 0.683.
- **causal:** Prior shift (I1-03), not spurious features, dominates the transfer error, which is why invariance over x does not help.
- **failed:** Invariance penalties do not improve transfer; dropping AlphaEarth hurts every design.
- **next:** Treat region-level prior explicitly rather than penalising risk variance.

## I3-03 - AlphaEarth proxy decomposition (corrected)

*Iteration 3 - Representation RSI - status: **baseline-candidate***

- **question:** Is AlphaEarth's value a re-encoding of named drivers or residual surface structure?
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)
- **experiment:** Each component alone: AE, AE_hat (ridge on Clim+Land+Human), AE_perp (residual)
- **results:** Drivers explain 11% of AE variance (linear). AUC block10 / LORO / zone: AE 0.751 / 0.624 / 0.685; AE_hat 0.691 / 0.603 / 0.622; AE_perp 0.625 / 0.534 / 0.555; Drivers+Human 0.713 / 0.656 / 0.658.
- **ecological:** AE's residual carries real in-region signal that does not transfer: local surface structure, or region identity (introduction history) leaking through.
- **causal:** AE acts as a high-capacity region fingerprint; it is not evidence for a transferable habitat mechanism.
- **failed:** The first version (I3-03a, Drivers+AE vs Drivers+AE_perp) was a linear reparameterisation and could not discriminate; superseded.
- **next:** Let AE dominate near data and fade under novelty (I4-01).

## I4-01 - Support-decay fusion with regime-matched training

*Iteration 4 - Spatial-Model RSI - status: **inconclusive***

- **question:** Do monotone support constraints and regime-matched inner OOF fix the weighting failure of I2-01?
- **hypothesis:** H4a: weights learned on inner OOF that includes leave-training-region-out transfer better. H4b: monotone decay (Geo weight exp(-d/rho); covariate experts fade with env novelty towards the prior) beats a free gate.
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)
- **architecture:** Log-linear pool of 6 experts (4 modality MLPs, all-feature MLP, kNN-geo) with ~15 parameters
- **validation:** block10, LORO, env, zone; 20 splits; paired sign tests
- **results:** Decay[mixed] vs free support gate: better in 17/20 splits on AUC and NLL (p = 0.003). vs Equal: 10/20 (p = 1.0). vs RF: 8/20. Learned correlation range rho: 7,600-22,000 km with block inner OOF, ~700 km with mixed; LORO Geo weight 0.32 -> 0.11, prior share 0.12 -> 0.56.
- **uncertainty:** LORO per-split differences range -0.04 to +0.13 AUC; 6 regions are too few to resolve gains < 0.02.
- **improved:** Monotone structure is reliably better than free gating; rho becomes ecologically interpretable when trained on the right regime.
- **failed:** No significant gain over equal pooling or RF; regime matching changes parameters but not accuracy (12/20).
- **next:** Fusion weighting is not the binding constraint; the regional prior is (I1-03).

## I5-01 - Exogenous species-pool covariate (Kraemer)

*Iteration 5 - Data RSI - status: **inconclusive***

- **question:** Does an independent coarse species-pool signal restore transfer?
- **hypothesis:** Gain under LORO/zone > gain under block10. Falsified if not.
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Base + log1p Kraemer compendium cells within 100 / 500 km for Ae. aegypti and Ae. albopictus
- **validation:** block10, LORO, zone, env
- **results:** MLP AUC gain: block10 +0.015, LORO +0.041, zone +0.019. RF: +0.007 / +0.010 / +0.002. LORO NLL barely changes (MLP 1.357 -> 1.344; RF 1.289 -> 1.314).
- **uncertainty:** MLP Europe hold-out AUC 0.526 (I1) vs 0.705 here with only seeds changed: neural extrapolation is seed-unstable.
- **improved:** Prediction holds for the MLP: transfer gain about 3x the in-distribution gain.
- **failed:** Calibration under transfer not fixed: the pool signal covers only 2 of 5 classes.
- **open questions:** Can pool information for Culex, Anopheles and 'other' be obtained (country checklists, GBIF non-iNat datasets)?
- **next:** Data RSI: species-pool layer for all classes; Spatial-Model RSI: pool term as the macro level of HiCo (see proposal P1).

## P1 - Proposal: Pool-Niche-Support (PNS) hierarchical model

*Iteration 6 - Spatial-Model RSI - status: **proposed***

- **question:** What architecture do the limitations that persist across iterations 1-5 justify?
- **hypothesis:** Composition = regional species pool (exogenous, partially pooled) x environmental niche (fused experts) x support-decayed spatial residual; HiCo coherence at the parent level.
- **architecture:** log lambda_s(x) = a_s(R(x)) + f_s(x) + k(d(x)) r_s(x); a_s: hierarchical regional-pool latent with a prior from exogenous checklists (Kraemer, GBIF non-iNat, WRBU); f_s: niche experts pooled with monotone novelty decay; r_s: spatial residual with learned correlation range rho; parent normalisation as HiCo.
- **validation:** LORO, zone, spatio-temporal with paired sign tests; RF and Equal pooling as the bars to beat
- **failed:** Not yet built; component evidence only (I1-03, I4-01, I5-01).
- **open questions:** Pool data for Culex/Anopheles/other; whether the regional prior can be estimated from few labelled cells in a new region (active sampling).
- **next:** I6: build pool layer for all classes, then the PNS ablation: drop a_s / drop decay / drop r_s / replace exogenous prior with uniform.

## M1 - Meta-lessons for the research strategy

*Iteration 6 - Scientific RSI - status: **infrastructure***

- **improved:** (1) Random and block CV hide the dominant failure; LORO + zone + oracle-prior decomposition are now standard outputs of every experiment. (2) Any learned weighting must be fitted on inner OOF whose support distribution matches deployment. (3) Neural extrapolation is seed-unstable: report >=5 seeds or per-seed spread for transfer claims. (4) A reparameterisation-equivalent ablation (I3-03a) is uninformative: check linear equivalence before running.
- **next:** Encode (1)-(3) in the shared experiment harness (rsi/common.py) before I6.

## I6-01 - Does HiCo's hierarchy cost accuracy? Coherence vs a strong flat model

*Iteration 6 - Spatial-Model RSI - status: **baseline-candidate***

- **question:** RF beats HiCo everywhere. Is that the price of coherence, or of HiCo's learned components?
- **hypothesis:** If coherence is HiCo's value, a post-hoc HiCo normalisation of RF (or HiCo with RF as the local term) keeps RF accuracy while being coherent.
- **data:** iNat research-grade Culicidae 2022-2025 (<=2 km), 7,004 report cells at 0.05 deg; 105 features (Climate 26, Landscape 11, Human 4, AlphaEarth 64)
- **representation:** Standardised raw covariates + AlphaEarth 2024 embedding (HiCo-Fuse store)
- **fusion:** HiCo normalisation eta = A(P) + B(x) - logmeanexp_P B, with B = RF log-ratios; A from an independent parent-level RF (RF+Recon) or learned through the fine likelihood (HiCo-RF)
- **architecture:** Arms: RF (separate fine and coarse models), RF-agg (coarse = average of fine), RF+Recon, HiCo (original), HiCo-RF; 2 seeds for neural arms
- **validation:** block10, LORO, env, zone (20 splits); scored at 0.05-deg cells and 0.5-deg parents (pooled held-out counts)
- **results:** Fine AUC block10/env/LORO/zone: RF 0.761/0.694/0.700/0.707; RF+Recon 0.761/0.696/0.686/0.703 (9/20 AUC wins, mean -0.004, p = 0.82); HiCo-RF 0.768/0.683/0.661/0.653 (6/20); HiCo 0.741/0.681/0.654/0.647 (2/20, p < 0.001). Separately fitted coarse and fine RF maps disagree: mean share TV 0.05-0.06, dominant class flips in 3.7-7.2% of parents. Coarse accuracy of RF-agg equals the separate coarse RF (12/20, p = 0.50).
- **error patterns:** HiCo-RF is the best arm under block10 (AUC 0.768, NLL 0.910 vs RF 0.930) and worse under LORO/zone: the jointly learned macro net fits the training geography (same mechanism as I2-01).
- **ecological:** A coarse map built as the average of the fine map is as accurate as a separately fitted coarse model, so a hierarchy only adds information when the coarse level has its own data (checklists, surveillance by admin unit).
- **causal:** None; this is a prediction-level result.
- **improved:** Coherence is free: post-hoc reconciliation removes all cross-scale contradiction at no significant accuracy cost.
- **failed:** HiCo as a learner: its neural local and macro terms are the source of the deficit, not the coherence constraint.
- **open questions:** Does a coarse term fed by data the fine model cannot see (species-pool checklists) beat RF-agg at the coarse level?
- **next:** Keep HiCo as the coherence operator; fill its macro term with the species-pool layer (P1), not with a learned net on parent-mean covariates.
