# Working causal hypotheses for reported mosquito composition

Written before the iteration-3 results were inspected. Revise via the ledger, not in place.

## Variables

| Node | Measured by | Role |
|---|---|---|
| C  Climate | WorldClim BIO1-19, ERA5 temperature, dew point, soil water, precipitation | Upstream driver (thermal and moisture limits on development, survival, overwintering) |
| L  Landscape / water | NDVI, EVI, forest, cropland, wetland, surface water, LST, elevation, slope | Driver and mediator (C -> L); larval habitat availability |
| H  Human settlement | WorldPop, built fraction, night lights, travel time | Habitat for container breeders **and** cause of observation |
| O  Observation process | not measured directly | Who reports, where, and which species are photographed |
| E  AlphaEarth embedding | 64-d annual embedding | Descendant of C, L, H and of unmeasured surface structure U |
| U  Fine surface structure | partly in E, partly in VLM concepts | Breeding sites (tyres, drains, informal housing), microclimate |
| Z  True local composition | unobserved | Target |
| R  Reports | iNaturalist counts per class | Z filtered by O |

## Edges assumed

C -> L, C -> Z, L -> Z, H -> L, H -> Z, H -> O, U -> Z, (C, L, H, U) -> E, Z -> R <- O.
Species introduction history I (ports, trade) -> Z, unmeasured and spatially structured.

## Identification statements

1. **Species-agnostic effort cancels.** If O scales every class equally at a site, the composition
   p(class | report, x) is unaffected by O. This is the target-group argument; it is an assumption.
2. **Species-specific detectability is not identified.** Conspicuous day-biting *Aedes* are more likely
   photographed than night-active *Culex* and *Anopheles*. That bias enters Z -> R and is confounded with H.
   No experiment here can separate it without survey data (AIMSurv, pending).
3. **E is a collider-adjacent proxy.** Conditioning on E mixes the effects of C, L, H and U. Its predictive value is
   not evidence that any of them is causal.
4. **Introduction history I is an unmeasured spatial confounder.** *Ae. albopictus* absence from suitable
   areas can reflect non-arrival. Spatially autocorrelated residuals and region-specific relations are
   expected consequences; region-invariant relations are more likely to be niche (C, L) effects.

## Testable implications (iteration 3)

- If C -> Z is a stable physiological relation, a climate-only expert should transfer across regions with
  a smaller drop than proxy-heavy experts (Human, E) - **transfer-matrix test**.
- If region-specific proxies drive in-region skill, invariance penalties (V-REx, group-DRO) should help
  leave-region-out and hurt in-region skill - **invariance test**.
- If E's value is mostly a re-encoding of C, L and H, replacing E with its residual given (C, L, H) should
  lose little; if U matters, the residual should keep the gain - **proxy decomposition**.

None of these tests identifies a causal effect. Each can falsify a specific causal reading of a predictive pattern.
