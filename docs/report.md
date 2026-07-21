# Landing-Site Selection near F2 (Faustini) — LOLA South-Pole Topography

**ISRO BAH PS-8 deliverable.** Propose a scientifically viable and SAFE landing site near a
doubly-shadowed crater (DSC) at the lunar south pole, from LOLA topography alone. Output: a
ranked landing point + safety polygon + multi-criteria suitability/hazard rasters, as a clean
hand-off to a (separate) path planner. No route is produced here.

---

## 1. Data

LOLA south-polar products (Barker et al. 2021), in south-polar-stereographic projection on a
sphere R = 1 737 400 m (MOON_ME / DE421), pixel-registered, 5 m/pixel:

- **DEM** `ldem_87s_5mpp.tif` — surface elevation Z (m).
- **Slope** `ldsm_87s_5mpp.tif` — surface slope (deg).

Full rasters are 40 000 × 40 000 px (±100 km). We **window-read only a ±15 km AOI** (30 km box,
6000 × 6000 px) around the science target — the full raster is never loaded.

**Target F2 (Faustini):** lat −87.39°, lon 82.31° → DEM CRS **X = 78 445.7 m, Y = 10 592.3 m**
(confirmed inside the DEM). F2 is doubly-shadowed; landers do not set down inside steep, dark
crater floors. The strategy is to land on a **safe, gentle bench near the rim** and let the rover
drive into F2.

AOI statistics: elevation min −3366.7 m / median −2123.1 m / max 496.3 m; slope median 11.2°,
max 70.1°. The high median slope means the safe set is selective — exactly the problem to solve.

## 2. Method overview

A transparent, physics-based, multi-criteria model (no ML, no training). Pipeline:

`00_prepare` (locate F2, crop AOI) → `01_terrain` (roughness/relief/curvature) →
`02_illumination` (horizon ray-cast → illumination index + PSR) → `03_suitability`
(normalize + weight + hazard) → `04_candidates` (3 profiles + landing-ellipse test +
quantified justification) → `05_figures` (figures + hand-off bundle).

All logic lives in `src/lunar_ice/`; scripts are thin CLI runners. Every raster carries CRS +
nodata; all parameters come from `config/config.yaml`.

## 3. Terrain derivatives (`01_terrain`)

Over a 50 m window (10 px):

| Layer | Definition | median | 90th pct | 99th pct |
|---|---|---|---|---|
| roughness | local std of slope (deg) — **BOULDER PROXY** | 1.33 | 2.46 | 4.26 |
| relief | local max−min elevation (m) | 11.38 | 20.69 | 30.24 |
| curvature | Laplacian of DEM (1/m) | ~0.000 | 0.0029 | 0.0371 |

**Caveat:** roughness is a *boulder proxy*. True boulder detection requires optical/OHRC imagery,
which is absent here; we use local slope variability as a stand-in for surface texture.

## 4. Illumination index + PSR (`02_illumination`) — the credibility upgrade

A **real horizon-based illumination index** computed from topography:

1. Block-mean the 5 m DEM to **20 m** (`compute_res_m`) for tractability (1500 × 1500).
2. For each of **72 azimuths**, compute the per-pixel **horizon elevation angle** by marching
   outward to **10 km** (`max_horizon_km`) and keeping the running max of
   `atan((z(d) − z0) / d)` — a vectorised shift-and-`fmax` sweep over the whole grid (dense-near /
   sparse-far step schedule), not a per-pixel double loop.
3. `illumination_index = mean_azimuth clip((sun_elev_max − horizon°) / sun_elev_max, 0, 1)`.

**Sun-elevation assumption:** at the lunar south pole the Sun never rises above
**~1.53°** (`sun_elev_max_deg`). A pixel whose horizon meets/exceeds that in every azimuth is never
directly lit → PSR (index ≈ 0); a pixel with a low horizon all around is lit a large fraction of
the year (index → 1). The index and `psr_mask` are resampled back to the 5 m grid.

**Result:** **6.23%** of the AOI is PSR. Sanity check confirms the physics — **F2's crater floor
comes out as PSR (index 0.000)** while rims/highs are lit: lowest-elevation decile (crater floors)
mean index 0.068 (32% PSR), highest-elevation decile (rims/highs) mean index 0.417 (0.4% PSR).

**Caveat:** this is an **annual-sunlit-FRACTION PROXY from topography vs the Sun-elevation cap —
not a modelled Sun ephemeris.** It captures where shadowing is topographically forced, which is
what matters for siting power/comms relative to PSRs.

## 5. Suitability + hazard (`03_suitability`)

**Criteria, each normalized to [0,1]:**

| criterion | sense | normalization |
|---|---|---|
| slope | low good | `1 − slope/10°` |
| roughness | low good | `1 − roughness/2.46°` (90th-pct threshold) |
| relief | low good | `1 − relief/20.69 m` (90th-pct threshold) |
| ice_proximity | nearer F2 better | ramps 1→0 over [rim+500 m, 8 km]; 0 inside |
| illumination | higher good | index from §4 |

**Hard hazard mask (no-go)** — a pixel is no-go if *any* of: slope > 10°, roughness > 90th pct,
relief > 90th pct, inside F2 rim + 500 m buffer, or beyond 8 km from F2. F2's rim radius is
estimated from the DEM (azimuthal-median elevation crest) at **580 m**, giving an inner exclusion
of **1080 m**.

**Go vs no-go: GO 14.51% / no-go 85.49%.** Dominant no-go reasons (overlapping): beyond 8 km
77.7% (an 8 km disc is only ~22% of the 30 km AOI — geometry), slope > 10° 55.3%.

**Weights** (`config.site_profiles`; same hard constraints, different soft weights):

| profile | slope | roughness | relief | ice_prox | illum |
|---|---|---|---|---|---|
| safest | 0.45 | 0.30 | 0.15 | 0.05 | 0.05 |
| closest_ice | 0.20 | 0.15 | 0.10 | 0.50 | 0.05 |
| best_lit | 0.20 | 0.15 | 0.05 | 0.10 | 0.50 |
| balanced | 0.30 | 0.25 | 0.15 | 0.20 | 0.10 |

## 6. The landing-ellipse test (`04_candidates`)

A credible site is a **contiguous flat patch big enough for a landing ellipse**, not one lucky
pixel. A pixel qualifies for the SAFE go-set only if a **full disc of `ellipse_radius_m` = 75 m
(15 px) around it is entirely within slope (≤10°) and roughness (≤2.46°) limits** — computed
exactly via a Euclidean-distance-transform erosion of the safety mask.

**13.18%** of the AOI passes the ellipse test; **4.59%** (1.65 M px) also lies in the F2 distance
band. For each profile, the top site is the **maximum mean-suitability over its 75 m ellipse**,
within the distance band, with sites kept non-overlapping.

## 7. The three sites + quantified justification

Every site reports numbers, not adjectives (metrics over its 75 m ellipse; lit% is the
illumination-index proxy; go-1km is % hazard-free terrain within 1 km):

| profile | rank | score | mean slope | max slope | rough pct | relief | dist F2 | lit% | go-1km |
|---|---|---|---|---|---|---|---|---|---|
| safest | 2 | 0.842 | 1.0° | 2.8° | 6th | 1.0 m | 1.82 km | 26% | 82% |
| **closest_ice** | **1** | **0.863** | 1.3° | 3.6° | 11th | 1.2 m | **1.38 km** | 18% | 67% |
| best_lit | 4 | 0.605 | 1.0° | 5.4° | 9th | 1.0 m | 5.32 km | **45%** | 97% |
| balanced | 3 | 0.793 | 1.5° | 3.7° | 9th | 1.5 m | 1.41 km | 22% | 67% |

**The trade-off (why three sites, not one winner):** all four sites are genuinely safe
(max slope ≤ 5.4°, local relief ≤ 1.5 m, all in the safe go-set). They differ by *objective*, not
safety:

- **closest_ice** — hugs F2 (1.38 km) for shortest rover traverse, but sits in dimmer terrain
  (18% lit).
- **best_lit** — more than doubles illumination (45%) and has the cleanest surroundings
  (97% go within 1 km), but pays for it at **5.3 km** from F2 (longer traverse).
- **safest** — flattest terrain (max slope 2.8°, 82% go), intermediate distance (1.8 km).
- **balanced** — a default compromise (1.4 km, 22% lit). **Recommended hand-off site.**

### Figure 1 — Hero landing map

![Candidate landing sites near F2 (Faustini)](../outputs/hero_landing_map.png)

AOI hillshade with the balanced suitability raster (green = best) over the go-areas, permanently
shadowed regions shaded cool blue, F2 outlined in white, and the top site of each objective marked
with the straight-line distance to F2. The recommended (balanced) site sits 1.41 km from F2 on a
gentle, well-connected bench.

### Figure 2 — Rover approach geometry (bench → rim → floor)

![Cross-section from the recommended site to the F2 floor](../outputs/cross_section_F2.png)

Topographic profile from the recommended landing bench to F2's centre: a flat bench (mean slope
1.5°), a crater rim crest (+18 m, ~13° wall), then the descent into the permanently shadowed floor
(the ice target). Total traverse 1.41 km, total descent ~142 m — the land-on-the-bench /
rover-drives-into-the-dark-floor concept made concrete. Slope-along-profile is on the right axis
(dashed line = the 10° safe limit).

## 8. Hand-off bundle (`outputs/`)

Per `docs/interface.md`, all on one shared grid/CRS (validated): 6000 × 6000 px, 5 m,
Moon2000_spole.

- **Rasters:** `suitability.tif` (balanced, [0,1]), `hazard.tif` (uint8 {0,1}), `slope.tif`
  (deg), `roughness.tif` (deg, boulder proxy), `illumination_index.tif` ([0,1] proxy),
  `psr_mask.tif` (uint8 {0,1}).
- **Vectors (lon/lat):** `landing_candidates.geojson` (4 ranked points + metrics),
  `landing_site_polygon.geojson` (75 m safety ellipse of the best site),
  `target_crater.geojson` (F2 outline, rim 580 m).
- **`manifest.json`** — per-layer path, description, units, CRS, pixel size, nodata.
- **Figures:** `hero_landing_map.png` (Figure 1), `cross_section_F2.png` (Figure 2),
  both 300 DPI, colourblind-safe.

The hand-off contains **no route, cost surface, or waypoints** — path planning is a separate
module's responsibility.

## 9. Caveats & assumptions (stated honestly)

- **No optical boulder detection.** Roughness (local slope std) is a *boulder proxy*; true
  boulder hazards need OHRC-class imagery not available here.
- **Illumination is a topographic proxy.** `illumination_index` is an annual-sunlit-fraction
  estimate from horizon angles vs a fixed max Sun elevation (~1.53°), not a modelled solar
  ephemeris or thermal model.
- **F2 rim radius is DEM-estimated** (azimuthal-median elevation crest, 580 m), not surveyed.
- **Single DEM source (LOLA).** No optical/stereo cross-validation; vertical accuracy per
  Barker et al. 2021.
- Vectors are written with lon/lat on the lunar sphere and tagged `EPSG:4326` to match the
  hand-off interface convention (the values are lunar geographic, not terrestrial WGS84).

## 10. Reference

Barker, M. K., Mazarico, E., Neumann, G. A., Smith, D. E., Zuber, M. T., & Head, J. W. (2021).
*Improved LOLA elevation maps for south pole landing sites: Error estimates and their impact on
illumination conditions.* Planetary and Space Science, 203, 105119.
https://doi.org/10.1016/j.pss.2020.105119
