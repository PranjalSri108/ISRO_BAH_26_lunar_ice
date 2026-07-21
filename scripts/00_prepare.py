"""Runner: 00_prepare — Locate F2 in DEM CRS and window-read the AOI. See docs/BUILD_PROMPTS.md.

Window-reads ONLY a +/-aoi.half_width_m box around F2 from the DEM and slope rasters,
writes dem_aoi.tif / slope_aoi.tif to data/interim with correct CRS/transform/nodata,
prints AOI bounds + shape + elevation/slope stats, and saves a hillshade PNG with F2 marked.
Never loads the full DEM.
"""
from __future__ import annotations

import numpy as np

import lunar_ice  # noqa: F401  (ensures package import works)
from lunar_ice import io_utils, viz


def _stats(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
        "nan_pct": 100.0 * (arr.size - finite.size) / arr.size,
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.transform import rowcol

    cfg = io_utils.load_config()
    target = cfg["target"]
    hw = float(cfg["aoi"]["half_width_m"])

    # F2 in DEM CRS, and the +/- hw box around it.
    fx, fy = io_utils.lonlat_to_xy(target["lon"], target["lat"], cfg)
    bounds = (fx - hw, fy - hw, fx + hw, fy + hw)
    print(f"F2 {target['name']}  lat={target['lat']} lon={target['lon']}  "
          f"-> X={fx:.1f} Y={fy:.1f} m")
    print(f"AOI half-width = {hw:.0f} m  ({2*hw/1000:.0f} km box)")

    interim = io_utils.resolve_path(cfg, cfg["outputs"]["interim"])

    layers = {}
    for key, out_name in (("dem", "dem_aoi.tif"), ("slope", "slope_aoi.tif")):
        src = io_utils.resolve_path(cfg, cfg["paths"][key])
        arr, profile = io_utils.window_read(src, bounds)
        out_path = interim / out_name
        io_utils.write_raster(out_path, arr, profile)
        layers[key] = (arr, profile)

        tl = profile["transform"] * (0, 0)  # top-left corner (x, y)
        br = profile["transform"] * (profile["width"], profile["height"])  # bottom-right
        s = _stats(arr)
        unit = "m" if key == "dem" else "deg"
        print(f"\n=== {key.upper()} -> {out_name} ===")
        print(f"  saved      : {out_path}")
        print(f"  shape      : {arr.shape[0]} rows x {arr.shape[1]} cols")
        print(f"  AOI bounds : left={tl[0]:.1f} top={tl[1]:.1f} "
              f"right={br[0]:.1f} bottom={br[1]:.1f}  (m)")
        print(f"  nodata     : {profile.get('nodata')}   dtype: {arr.dtype}")
        print(f"  stats ({unit}) : min={s['min']:.3f} median={s['median']:.3f} "
              f"mean={s['mean']:.3f} max={s['max']:.3f} std={s['std']:.3f} "
              f"nan={s['nan_pct']:.2f}%")

    # Hillshade PNG with F2 marked, to confirm the crater is in-frame.
    dem_arr, dem_profile = layers["dem"]
    hs = viz.hillshade(dem_arr, pixel_size_m=cfg["crs"]["pixel_size_m"])
    row, col = rowcol(dem_profile["transform"], fx, fy)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(hs, cmap="gray", origin="upper")
    ax.scatter([col], [row], s=140, facecolors="none", edgecolors="red", linewidths=2)
    ax.annotate(target["name"], (col, row), color="red", fontsize=12,
                xytext=(10, 10), textcoords="offset points")
    ax.set_title(f"AOI hillshade ({2*hw/1000:.0f} km) — {target['name']} marked")
    ax.set_xlabel("col (px)")
    ax.set_ylabel("row (px)")
    png_path = interim / "hillshade_aoi.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    in_frame = 0 <= row < dem_arr.shape[0] and 0 <= col < dem_arr.shape[1]
    print(f"\nHillshade PNG : {png_path}")
    print(f"F2 pixel (row,col) = ({row},{col})  in-frame? {in_frame} "
          f"-> {'VISIBLE' if in_frame else 'OUT OF FRAME'}")


if __name__ == "__main__":
    main()
