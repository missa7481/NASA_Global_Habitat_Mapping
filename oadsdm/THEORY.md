# Identification sketch for OA-DSDM

## Setup
Covariates x = (h, e); e ⊂ h are human-footprint/effort covariates (population, built-up,
night lights, travel time). For a peridomestic vector, e is *both* habitat and observation effort.

* Opportunistic target vs. target-group background (TGB), conditional on a culicid report at x:
  `logit P(target | report, x) = η(x) + g(e) + c` where g is the target-specific reporting
  propensity relative to the TGB (e.g. a conspicuous day-biter photographed more in cities).
* Survey source s: occupancy `z ~ Bern(σ(η(x) + α_s))`, per-visit detection
  `p_sj = σ(δ_s + κ_s·log effort_sj)` if z = 1.

## Claim 1 (non-identification from PO/TGB alone)
For any function φ(e), (η + φ, g − φ) gives the same PO likelihood. A PB classifier therefore
estimates η + g: the habitat effect of urbanness and the reporting effect are inseparable.
A flexible h-encoder (e.g. AlphaEarth, from which e is recoverable with spatial-CV R² ≈ 0.82–0.88
in this cohort) can express any such φ through h.

## Claim 2 (identification with surveys spanning e)
With repeated-visit survey data, (η + α_s) is identified on the survey covariate support S_s
(standard occupancy identification with ≥2 visits and detection varying with effort). Then
g(e) = [η + g](x) − η(x) is identified for e ∈ proj_e(S_s) ∩ proj_e(S_PO), and η is identified up
to a constant on the PO support wherever the needed e-values are covered by some survey.

## Consequence (testable; the semi-synthetic benchmark checks it)
* If surveys cover only low-footprint sites (NEON: median log-population 0, built fraction 0),
  then proj_e(S_s) is a point and g is not identified: OA-DSDM ≈ PB on urban sites.
* If surveys span the footprint gradient (Austria, AIMSurv), OA-DSDM should remove the effort
  leakage that PB carries into η on urban sites, without losing the genuine habitat effect of
  urbanness.
* Where survey coverage is missing, the remaining ambiguity is localised in e-space. That
  is what makes "where should the next survey be?" a well-posed question (an active-design
  objective), which is the practical NASA deliverable.

## Assumptions to state honestly
1. TGB reports share the target's spatial effort surface up to the function g(e) (no
   residual effort varying with non-e habitat covariates).
2. The survey occupancy and PO suitability share η up to a constant (same species, and
   compatible years: 2024 PO vs 2024/2025 surveys; AIMSurv 2020 is older).
3. Detection is conditionally independent across visits given occupancy and effort.
