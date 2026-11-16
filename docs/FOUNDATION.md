> Archived README of the foundation stage (single-species *Ae. albopictus* presence-background benchmark). Paths such as `..\webapp` and `..\.venv` refer to the original research workspace and its frozen cohort, which are not part of this public repository.

# Global Mosquito Habitat Foundations

This project is the data and representation-learning foundation for an interactive global mosquito habitat explorer. It starts from the reproducible inputs and frozen results described in `reports\Global_Habitat_Project_Complete_Report_2026_11_06.html`, while keeping all source observations and large raster products in the shared research workspace.

## Research objective

For *Aedes albopictus*, evaluate whether ecological covariates, satellite-derived representations, and their combinations improve spatially transferable habitat-suitability ranking. Keep occurrence, detection, abundance, and disease risk as separate outcomes. Maps are retrospective research estimates, not calibrated occurrence probabilities.

The report's latest frozen development cohort has 1,601 locations (739 presences and 862 target-group background locations), 41 engineered environmental predictors, and 64 annual AlphaEarth embedding dimensions. The background locations are not confirmed absences. The report's selected AlphaEarth64 + logistic-regression model reached mean spatial-fold AP 0.811; Environment41 + AlphaEarth105 + logistic regression reached 0.810. Their small difference is not evidence that one representation is superior. Existing external survey results are diagnostics, not a new blind global validation.

## Why *Aedes albopictus*

The report already has the most complete compatible multi-source cohort and aligned feature extraction for this invasive, medically important vector. It supports a meaningful comparison without pooling different species or incompatible survey labels. The initial scope is global retrospective habitat suitability at the 2024 reference year; expansion to other species requires separate label and evaluation protocols.

## Data inventory and roles

| Data family | Products / candidate sources | Use and limits |
|---|---|---|
| Target observations | iNaturalist records obtained through GBIF; 2024 Austrian AGES surveillance | Main occurrence cohort after coordinate-quality screening, duplicate resolution, and source tracking. GBIF is an aggregator, not a separate field survey. |
| Independent or diagnostic observations | NEON mosquito data product DP1.10043.001; Mosquito Alert; GLOBE Observer; Austrian 2025 survey | Keep separate by survey method, year, and label semantics. Previously inspected endpoints must not be described as blind validation. GLOBE habitat reports do not establish target-species presence. |
| Additional public occurrence candidates | VectorMap / VectorBase and published species compilations | Screen source, taxon, date, spatial precision, effort, license, and overlap before use; county records and coarse locations are not point observations. |
| Abundance candidates | NEON trap/count records, Austrian trap effort and counts, and the repository's historical VectAbundance archive | Potential regional follow-up only. Trap type, duration, season, life stage, and spatial grain differ; VectAbundance is historical and coarse (about 9 km). Do not merge raw counts into a global abundance target. |
| Climate and terrain | WorldClim v2.1 BIO19 (1970–2000); ERA5-Land 2024; MERIT elevation and slope | Long-term climate and annual weather are distinct time scales. Native resolutions remain coarse even when summarized to a finer output grid. |
| Surface and human context | MODIS NDVI/EVI, land cover and land-surface temperature; JRC surface water; WorldPop 2024, GHSL 2020, VIIRS 2024, accessibility 2015 | Useful habitat proxies, but not container-level breeding habitat or uniform global measurements. Coverage and product years differ. |
| Learned satellite representation | Google / DeepMind AlphaEarth annual 64D embedding, 2024 | Treat all dimensions jointly as learned features, not independently interpretable physical variables. Global/polar masks and possible pretraining overlap limit claims. |

Primary official product references: [GBIF](https://www.gbif.org/), [WorldClim v2.1](https://worldclim.org/data/worldclim21.html), [ERA5-Land in Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR), [AlphaEarth annual embedding](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL), [NEON DP1.10043.001](https://data.neonscience.org/data-products/DP1.10043.001), [GLOBE Observer mosquito habitat data](https://observer.globe.gov/get-data), and [Mosquito Alert](https://www.mosquitoalert.com/).

### Expanded source coverage and admission decisions

The 2026-11-12 source audit adds geographic context without pooling incompatible labels:

- The development cohort remains 1,601 locations (739 presences, 862 target-group background) across 67 countries.
- GBIF's 2024 screen found 1,278 *Ae. albopictus* records, of which 720 passed a <=500 m precision screen; all 720 were already in the local archive. The 2026 screen found 1,646 records, including 1,014 passing the same screen. These new-year records remain reserved presence-only candidates, not survey negatives or training labels; a frozen GBIF DOI is still missing.
- Mosquito Alert's 2024 release contains 4,061 high-confidence reports, including 263 IDs absent from the older archive. The selected 600-location candidate subset spans 21 nonblank country codes. Numeric location uncertainty is missing, so it remains exploratory and outside training. The same release reports 3,630 high-confidence 2025 reports; those are not mixed into the 2024 cohort.
- GLOBE Observer contributes 12,316 mosquito-habitat reports from 2023-2026; 12,306 have valid coordinates across 74 countries. There are 48 coordinate-valid species claims in this archive and none are expert-verified. They are habitat observations, not *Ae. albopictus* labels.
- NEON contributes 373 2026 sites more than 10 km from development locations, with 24 detections. This U.S.-focused temporal survey is kept as an external diagnostic, not used to fit or select models.

The public coverage map aggregates these roles into 5-degree cells. It does not publish point coordinates. Candidate record volume is not equivalent to independent geographic training coverage.

### Main gaps

- Sampling remains strongly clustered in North America and Europe; new independent tropical surveys are needed.
- Presence-background labels reflect both ecology and observation effort. A background point is not a verified absence.
- A formal frozen GBIF download DOI, complete source-overlap audit, and globally independent validation are still required.
- Survey methods, trap effort, identification confidence, and coordinate uncertainty are inconsistent across sources.
- Climate, satellite, population, and land-cover products have differing reference years, spatial supports, and coverage masks.
- No globally standardized, spatially dense abundance dataset is currently suitable for a continuous *Ae. albopictus* abundance map. Abundance is therefore a separately scoped regional research question.

## Representation plan

1. **Ecology-informed features:** preserve the frozen 19 bioclimatic variables and 41-variable environmental recipe, with units, native resolution, year, and quality masks. These are interpretable and establish the conventional SDM baseline.
2. **Linear combinations:** compare climate/environment-only, AlphaEarth-only, and early concatenation (`Environment41 + AlphaEarth64`) using the same spatial folds and regularized logistic regression. A train-only PCA variant can test whether reducing the embedding dimension improves stability; it is a new experiment, not part of the frozen results.
3. **Nonlinear combinations:** compare single-modality MLPs, early-fusion MLP on the 105 concatenated features, and a nonlinear late-fusion stack. The late-fusion meta-MLP is trained on four-fold, 5-degree spatial-grouped out-of-fold predictions from separate environment and embedding MLPs. All scaling is fitted inside training partitions. These are shallow downstream learners over pretrained embeddings, not new foundation-model pretraining.
4. **Evaluation:** compare fold-wise AP and ROC-AUC against the frozen six-fold MaxNet, GLM and Random Forest baseline results. Survey detection results remain separate. Do not compare scores across different cohorts or label mechanisms as if they were the same endpoint.

The early-fusion MLP reached AP 0.815 (fold SD 0.037), versus 0.810 (0.045) for the AlphaEarth-only MLP; the nonlinear late-fusion stack reached 0.786 (0.072). These descriptive results do not establish a reliable gain from fusion. The MLP is not deployed as a new global surface. Existing global predictions remain in the public explorer.

## Build the reproducible foundation bundle

Run from this directory with the shared workspace available:

```powershell
& '..\.venv\Scripts\python.exe' 'scripts\build_foundation_bundle.py'
& '..\.venv\Scripts\python.exe' 'scripts\run_fusion_benchmark.py'
```

The bundle builder reads the frozen cohort and the source-audit files under `..\webapp\runtime\candidates\admission_2026_11_12\`; it writes local artifacts under `artifacts\`. The MLP experiment uses scikit-learn and the same frozen outer spatial contexts. Override locations with `--source-root`, `--output-dir`, or `--artifact-dir`.

To publish only reviewed aggregate layers and metrics to the sibling GitHub Pages repository:

```powershell
& '..\.venv\Scripts\python.exe' 'scripts\build_foundation_bundle.py' --public-data-dir '..\github_site\data'
& '..\.venv\Scripts\python.exe' 'scripts\run_fusion_benchmark.py' --public-data-dir '..\github_site\data'
```

Outputs:

- `artifacts\dataset\site_features.csv` — frozen primary sites, labels, source/effort fields, and the 105 declared predictors. This is a local research derivative; review licensing and coordinate privacy before any public release.
- `artifacts\dataset\coverage_5deg.geojson` — coarse aggregated training-data coverage, not exact site coordinates.
- `artifacts\dataset\source_coverage_5deg.geojson` and `source_inventory.json` — separate 5-degree coverage and admission status for training, candidate, auxiliary-habitat and external-survey sources.
- `artifacts\results\representation_comparison.csv` — frozen development endpoint metrics for the four declared representations and available lightweight learners.
- `artifacts\results\paired_representation_deltas.csv` — fold-paired descriptive AP differences for common GLM/RF comparisons; fold spread is not an independent-sample confidence interval.
- `artifacts\results\fusion_mlp_fold_metrics.csv`, `fusion_mlp_summary.csv`, and `fusion_mlp_protocol.json` — same-context shallow MLP ablations, including fold-level results and preprocessing/model provenance.
- `artifacts\provenance.json` — source and output SHA-256 checksums plus cohort/feature counts.

Run the focused checks:

```powershell
& '..\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

## Website stages

The GitHub Pages project at `..\github_site\` now links to an interactive research page with source-specific coverage, admission caveats, matched model results and an annotated methods bibliography. The existing explorer remains the source for global suitability and quality maps; the MLP experiment is not yet a global prediction product. Public data outputs include aggregate polygons and fold metrics only, never site coordinates, raw observations or site-level predictions.

## Representation-learning papers

- [SatMAE (Cong et al., 2022)](https://arxiv.org/abs/2207.08051) — temporal and multispectral satellite masked-autoencoder pretraining.
- [AnySat (Astruc et al., 2024)](https://arxiv.org/abs/2412.14123) — self-supervised multimodal EO representations across varied scales and sensors; arXiv preprint.
- [Prithvi-EO-2.0 (Szwarcman et al., 2024)](https://arxiv.org/abs/2412.02732) — large-scale multitemporal EO foundation model; arXiv preprint.
- [TerraMind (Jakubik et al., 2025)](https://arxiv.org/abs/2504.11171) — generative multimodal EO; arXiv version.
- [OCELOT (van Merriënboer et al., 2026)](https://arxiv.org/abs/2611.00284v2) — planetary-scale relative-occurrence modeling with satellite embeddings; recent preprint and not evidence of equivalent mosquito-scale gains.
- [Zbinden et al. (2024)](https://doi.org/10.1016/j.ecoinf.2024.102623) — pseudo-absence selection and spatial evaluation for deep SDMs.
- [Kellenberger et al. (2026)](https://doi.org/10.1111/geb.70184) — deep-learning versus conventional SDM performance and leakage-aware validation.
- [McMahon et al. (2022)](https://doi.org/10.1093/jme/tjac145) — satellite and ground-based environmental suitability measurements for vector mosquitoes, including *Ae. albopictus*.
