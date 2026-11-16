# Global Mosquito Habitat Mapping with Earth-Observation Fusion

Model development and benchmarks for global mosquito (Culicidae) habitat mapping. The models fuse climate, landscape, human-footprint and AlphaEarth satellite-embedding representations, and every one is tested for spatial transfer.

**Project site:** https://missa7481.github.io/NASA_Global_Habitat_Mapping/ · **Benchmark tables:** [BENCHMARK.md](BENCHMARK.md) · **Research ledger:** [rsi/memory/RESEARCH_MEMORY.md](rsi/memory/RESEARCH_MEMORY.md)

> Independent research prototype. Not an official NASA product.

## Model families

Four model families were developed in order. Each started from the failures of the one before.

| Stage | Folder | Task | Models | Validation |
|---|---|---|---|---|
| 1. Foundation | `scripts/` | *Ae. albopictus* presence vs target-group background (1,601 sites) | GLM, MaxNet, BRT, RF, MLP. Representations: BioClim19, Environment41, AlphaEarth64, early and late fusion | 6 frozen spatial folds |
| 2. Observation-aware SDM | `oadsdm/` | Shared suitability with separate observation processes: opportunistic reports with an effort-bias head, and repeat-visit surveys | PB, Pooled, SurveyOnly, ISDM, **OA-DSDM** (linear and neural); semi-synthetic identifiability benchmark | Frozen protocol ([PROTOCOL](oadsdm/PROTOCOL.md), [THEORY](oadsdm/THEORY.md)); NEON 2026 and Austrian surveys |
| 3. HiCo-Fuse | `hicofuse/` | Multinomial composition of iNaturalist Culicidae reports (5 classes) per 0.05° cell, 7,004 cells, 105 features | Linear and MLP early fusion, single-modality, gated mixture of experts, **HiCo** (coarse-to-fine hierarchy that keeps maps coherent), HiCoMoE, two-part support model; VLM concept probes | 10° spatial blocks; external Kraemer and NEON checks |
| 4. Research loop | `rsi/` | The same composition task, with a recursive cycle: observe, diagnose, hypothesise, experiment, record | 9 baselines (Prior, kNN-geo, GLM, MLP, MLP+coords, HGB, RF, GatedMoE, HiCo); 8 weighting schemes; invariance (V-REx, group DRO); support-decay fusion; species-pool covariates; coherence reconciliation | 9 holdout designs: random, 5°/10°/20° blocks, leave region out, environmental cluster, climate zone, temporal, space-time |

### Main findings

- **Random splits overstate skill.** Holding out a whole region costs the random forest 0.10 AUC and pure geographic interpolation 0.26.
- **The random forest is the strongest model on every holdout design.**
- **Most transfer error is the regional species mix, not the habitat relationship.** Re-weighting predictions to the true regional class mix brings the forest's held-out log-loss from 1.30 to 1.07.
- **Learned fusion weights fit the training geography.** Monotone support-decay fusion beats a free attention gate in 17 of 20 splits.
- **Coherence costs nothing.** Applying HiCo's normalisation to any model after training removes every cross-scale contradiction with no measurable loss of accuracy. A hierarchy adds information only when the coarse level has data of its own.
- **AlphaEarth can encode reporting effort.** It predicts human-footprint covariates with spatial-CV R² up to 0.88, so presence-background models can learn where people report, not only where mosquitoes live.

## Repository layout

```
index.html, assets/      project site (GitHub Pages)
BENCHMARK.md             all benchmark tables, generated from artifacts/
hicofuse/                HiCo-Fuse data store, models (models.py), training, multi-resolution, maps, VLM probes
oadsdm/                  observation-aware integrated SDM: data blocks, likelihoods, theory and protocol
rsi/                     research loop: common.py (holdout designs, metrics, baselines), it1–it6 experiments,
                         CAUSAL_DAG.md, memory/ (structured ledger), report and site builders
scripts/                 foundation bundle, fusion benchmark, OA-DSDM runner, semi-synthetic benchmark, feature extraction
tests/                   unit tests
artifacts/               aggregate results only (fold and split metrics, summaries)
docs/FOUNDATION.md       archived README of the foundation stage (data inventory, representation plan)
```

## Reproducing

Python 3.11+ with a CUDA build of PyTorch. Install with `pip install -r requirements.txt`.

```bash
python -m unittest discover -s tests -v
python rsi/it1_baselines.py         # nine baselines x nine holdout designs
python rsi/it6_coherence.py         # coherence vs accuracy
python rsi/build_report.py          # report from artifacts/
python rsi/publish_repo.py          # regenerate BENCHMARK.md
```

The experiments need the HiCo-Fuse feature store (`artifacts/hicofuse/dataset.npz` and `store/`), built by `hicofuse/build_store.py` from the iNaturalist download (`hicofuse/get_occurrences.py`) and the global covariate rasters. Stages 1–2 read a frozen cohort and Earth Engine extraction modules from the original research workspace. Those are not included. `hicofuse/vlm_chips.py` needs an Earth Engine project: set `EE_PROJECT`.

## Data policy

This repository publishes only aggregate metrics and 0.2° model maps. It does not publish observation coordinates, site-level features, per-site or per-cell predictions, or raw downloads.

**Data sources:**
- iNaturalist research-grade observations via GBIF
- WorldClim v2.1, ERA5-Land, MODIS, JRC surface water, WorldPop, GHSL and VIIRS
- Google AlphaEarth Foundations annual embeddings (2024)
- Kraemer et al. (2015) *Aedes* compendium
- NEON DP1.10043.001
- Austrian AGES surveillance

Each source keeps its own license and terms.
