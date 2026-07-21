# Figure guide

Every figure below is reproducible from the pipeline; the versions in
[`../assets/`](../assets) are the ones embedded in the README. Rasters that back them
(`outputs/*.tif`) are git-ignored — regenerate them by running the pipeline
(see the README quickstart).

| Asset | Produced by | What it shows |
|-------|-------------|---------------|
| `assets/pipeline_hero.png` | `scripts/09_pipeline_figure.py` | The end-to-end story in three panels — **detect → land → traverse** — on real Faustini-F2 data. |
| `assets/hero_landing_map.png` | `scripts/05_figures.py` | LOLA hillshade of the ±15 km AOI with the suitability overlay, the four ranked landing candidates, and the F2 outline. |
| `assets/cpr_dop_scatter.png` | `scripts/08_fp_detect.py` | CPR-vs-DOP density for the full-pol scene; the `CPR > 1 & DOP < 0.13` quadrant isolates the volume-scattering (subsurface-ice) pixels. |
| `assets/dfsar_validation_overlay.png` | `scripts/06_dfsar_detect.py` | DFSAR L3C-derived-mosaic ice mask over the AOI, overlaid on CPR, with the ICY_CRATERS_SP catalogue for cross-checking. |
| `assets/headline_threshold_350x.png` | analysis figure | The inherited-threshold audit: `DOP < 0.87` floods in surface scatter (166 km²) versus `DOP < 0.13` (0.47 km²) — a ~350× over-detection. |
| `assets/screenshot_*.png` | slide captures | Presentation stills of the detect / land / traverse stages. |

## Regenerating

```bash
# figures depend on the raster/vector outputs of the pipeline stages
PYTHONPATH=src python3 scripts/05_figures.py         # landing figures + hand-off
PYTHONPATH=src python3 scripts/09_pipeline_figure.py  # three-panel hero
```

Outputs land in `outputs/` (git-ignored); copy the ones you want to publish into
`assets/`.
