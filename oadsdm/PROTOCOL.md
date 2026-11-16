# OA-DSDM evaluation protocol (frozen 2026-11-14, before any OA-DSDM fit)

## Question
Does jointly modelling the ecological process (shared latent suitability) and
source-specific observation processes (opportunistic reporting bias; survey
detection with effort) improve transfer to *effort-controlled survey outcomes*
relative to the presence-background (PB) classifiers used so far, while using
the same features?

## Data blocks (never pooled into one binary label)
| Block | Rows | Role |
|---|---|---|
| iNaturalist *Ae. albopictus* vs other-Culicidae target-group background | 625 / 862 | PO thinned-process likelihood |
| Austria 2024 ovitrap monitoring (n events, k positive) | 313 sites | survey occupancy-detection likelihood |
| NEON 2025 trap events (trap-hours per event) | 438 sites, 8,939 events, 46 clusters | survey occupancy-detection likelihood (train clusters only) |
| NEON 2026 annual detection | 373 primary sites (>10 km from development) | **test only**, held-out clusters |

## Splits
* PO + Austria: the six frozen outer contexts (`contexts.json`, seeds 42/202 x 3 folds, ~20 km buffer).
* NEON: siteID clusters (union of 2025 and 2026) are partitioned into 3 folds per seed
  (seed 42 -> contexts s42_f*, seed 202 -> s202_f*). Clusters with any 2025 or 2026 detection
  are dealt round-robin first so every fold contains positives. In context s_f the clusters
  of fold f are test: their NEON 2025 data are removed from training and their NEON 2026
  labels are scored. Every NEON cluster is therefore scored exactly once per seed.
* All scaling is fitted on the training rows of each context.

## Endpoints
* **Primary:** NEON 2026 held-out-cluster pooled out-of-fold ROC-AUC and AP (per seed, then mean
  of the two seeds); 1,000-draw siteID-cluster bootstrap for paired model differences.
* Secondary: PB AP on frozen test folds; Austrian survey AUC on test folds (detected vs not).
* NEON 2026 was inspected in earlier project stages, so results are a diagnostic, not a blind
  validation. No hyperparameter is chosen using any test endpoint.

## Models (identical features, folds, optimiser, epochs and seeds)
* `PB` — PO likelihood only (= current practice, neural).
* `Pooled` — PO + survey sites as extra binary labels (detected=1 / not=0) in one logistic loss.
* `SurveyOnly` — survey occupancy-detection likelihood only.
* `ISDM-noBias` — PO + survey likelihoods, shared suitability, no PO bias head.
* `OA-DSDM` — PO + survey likelihoods, shared suitability, PO bias head g(e) on effort covariates.
* Linear (`eta` linear) versions of `PB` and `OA-DSDM` as the classical ISDM reference.

## Fixed hyperparameters (chosen a priori)
Suitability MLP: 2 hidden layers x 64, SiLU, dropout 0.1. Bias head: 1 hidden layer x 16.
AdamW lr 3e-3, weight decay 1e-3, 400 full-batch epochs, cosine schedule; 5-member seed
ensemble (mean of standardized eta). Equal weighting of per-source mean negative log-likelihoods.
Representation for the main comparison: Environment41 + AlphaEarth64 (105).
