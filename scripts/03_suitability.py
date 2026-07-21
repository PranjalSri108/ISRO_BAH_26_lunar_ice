"""Runner: 03_suitability — Normalize criteria, hazard mask, per-profile suitability. See docs/BUILD_PROMPTS.md.

Normalize slope/roughness/relief (low good), ice_proximity (nearer F2 better, crater interior
excluded), illumination (02 output, higher good) to [0,1]. Build hazard.tif (hard no-go), then a
suitability_<profile>.tif per config.site_profiles (weighted sum, =0 inside hazard).
Prints %go vs no-go and per-profile suitability stats over go-areas.
"""
from __future__ import annotations

import numpy as np

import lunar_ice  # noqa: F401
from lunar_ice import io_utils, suitability as su


def _read(interim, name):
    import rasterio
    with rasterio.open(interim / name) as ds:
        return ds.read(1), ds.profile.copy()


def main() -> None:
    cfg = io_utils.load_config()
    interim = io_utils.resolve_path(cfg, cfg["outputs"]["interim"])

    slope, profile = _read(interim, "slope_aoi.tif")
    dem, _ = _read(interim, "dem_aoi.tif")
    roughness, _ = _read(interim, "roughness.tif")
    relief, _ = _read(interim, "relief.tif")
    illum, _ = _read(interim, "illumination_index.tif")
    transform = profile["transform"]

    # F2 geometry: centre, crater rim radius, distance field.
    fx, fy = io_utils.lonlat_to_xy(cfg["target"]["lon"], cfg["target"]["lat"], cfg)
    R_f2 = su.estimate_crater_radius(dem, transform, fx, fy)
    buf = float(cfg["landing"]["min_dist_from_F2_rim_m"])
    inner_excl = R_f2 + buf
    max_dist = float(cfg["landing"]["max_dist_from_F2_m"])
    dist = su.distance_to_point(dem.shape, transform, fx, fy)
    print(f"F2 rim radius (est) = {R_f2:.0f} m  -> inner exclusion = rim+{buf:.0f} "
          f"= {inner_excl:.0f} m ; outer = {max_dist:.0f} m from F2")

    # Hazard mask + the thresholds used.
    hazard, thr = su.build_hazard(slope, roughness, relief, dist, inner_excl, cfg)
    io_utils.write_raster(interim / "hazard.tif", hazard,
                          {**profile, "dtype": "uint8", "nodata": None})
    print("\nHAZARD thresholds:")
    for k, v in thr.items():
        print(f"  {k:22s}: {v:.4g}")
    go = hazard == 0
    n = hazard.size
    print(f"\nGO area : {100*go.mean():.2f}%   NO-GO : {100*(1-go.mean()):.2f}%  "
          f"({go.sum():,} / {n:,} px)")
    # Per-reason no-go breakdown (overlapping).
    reasons = {
        "slope>max": slope > thr["slope_max_deg"],
        "rough>max": roughness > thr["roughness_max"],
        "relief>max": relief > thr["relief_max_m"],
        "inside rim+buf": dist < inner_excl,
        "beyond maxdist": dist > max_dist,
    }
    print("no-go reasons (overlapping, % of AOI):")
    for k, m in reasons.items():
        print(f"  {k:16s}: {100*m.mean():.2f}%")

    # Normalized criteria in [0,1].
    criteria = {
        "slope": su.normalize_low_good(slope, thr["slope_max_deg"]),
        "roughness": su.normalize_low_good(roughness, thr["roughness_max"]),
        "relief": su.normalize_low_good(relief, thr["relief_max_m"]),
        "ice_proximity": su.ice_proximity(dist, inner_excl, max_dist),
        "illumination": np.clip(illum, 0.0, 1.0),
    }

    # Per-profile suitability.
    print("\nPer-profile suitability over GO-areas (mean / 90th / max):")
    for pname, weights in cfg["site_profiles"].items():
        s = su.weighted_suitability(criteria, weights, hazard)
        io_utils.write_raster(interim / f"suitability_{pname}.tif", s, profile)
        vals = s[go]
        print(f"  {pname:12s} mean={vals.mean():.3f}  "
              f"90th={np.percentile(vals,90):.3f}  max={vals.max():.3f}")

    print(f"\nsaved hazard.tif + suitability_<profile>.tif ({len(cfg['site_profiles'])}) "
          f"to {interim}")


if __name__ == "__main__":
    main()
