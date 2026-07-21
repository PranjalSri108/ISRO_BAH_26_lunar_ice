"""Runner: 01_terrain — Roughness / local relief / curvature over the AOI. See docs/BUILD_PROMPTS.md.

From dem_aoi (+ slope_aoi): roughness = local std of slope in rough_win_m (a BOULDER PROXY —
no optical/OHRC here), local relief = max-min elevation in that window, curvature = Laplacian.
Saves roughness.tif / relief.tif / curvature.tif and prints median/90th/99th pct of each.
"""
from __future__ import annotations

import numpy as np

import lunar_ice  # noqa: F401
from lunar_ice import io_utils, terrain


def _pcts(arr: np.ndarray) -> tuple[float, float, float]:
    finite = arr[np.isfinite(arr)]
    return tuple(np.percentile(finite, [50, 90, 99]))


def main() -> None:
    import rasterio

    cfg = io_utils.load_config()
    interim = io_utils.resolve_path(cfg, cfg["outputs"]["interim"])
    px = float(cfg["crs"]["pixel_size_m"])
    win = terrain.win_px(float(cfg["terrain"]["rough_win_m"]), px)
    print(f"rough_win_m={cfg['terrain']['rough_win_m']}  pixel={px}m  -> window={win}px")

    with rasterio.open(interim / "dem_aoi.tif") as ds:
        dem = ds.read(1)
        profile = ds.profile.copy()
    with rasterio.open(interim / "slope_aoi.tif") as ds:
        slope = ds.read(1)

    layers = {
        "roughness": terrain.roughness(slope, win),
        "relief": terrain.local_relief(dem, win),
    }
    if cfg["terrain"].get("curvature", True):
        layers["curvature"] = terrain.curvature(dem, px)

    units = {"roughness": "deg (slope-std)", "relief": "m", "curvature": "1/m"}
    for name, arr in layers.items():
        out_path = interim / f"{name}.tif"
        io_utils.write_raster(out_path, arr, profile)
        p50, p90, p99 = _pcts(arr)
        print(f"\n=== {name}.tif  ({units[name]}) ===")
        print(f"  saved  : {out_path}")
        print(f"  median : {p50:.4f}   90th : {p90:.4f}   99th : {p99:.4f}")
        if name == "roughness":
            print("  NOTE   : roughness is a BOULDER PROXY (no optical/OHRC imagery here).")


if __name__ == "__main__":
    main()
