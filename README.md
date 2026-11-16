# Vector Habitat Research Log

What breaks when a mosquito habitat model leaves home: a recursive research loop on global mosquito habitat mapping with Earth-observation data fusion (climate, landscape, human footprint and AlphaEarth satellite embeddings).

Open `index.html`, or browse the GitHub Pages site for this repository.

- Six iterations, tested under nine holdout designs (random, spatial blocks, leave one region out, environmental cluster, climate zone, temporal, space-time).
- Main finding: transfer to new regions fails mainly because each region has a different mix of species. Neither the fusion layer nor the global-to-local hierarchy is the cause.
- Iteration 6: applying HiCo's coherence rule after training keeps random-forest accuracy and removes every contradiction between scales.

Every number on the page is generated from experiment output files. The page publishes only aggregate results and 0.2° model maps, never observation coordinates.

**Data sources:**
- iNaturalist research-grade Culicidae observations, 2022–2025, via GBIF
- Bioclimatic, landscape and human-footprint layers
- Google AlphaEarth Foundations embeddings (2024)
- Kraemer et al. (2015) *Aedes* compendium

This is an independent research prototype, not an official NASA product.
