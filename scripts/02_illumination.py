"""Runner: 02_illumination — Horizon-based illumination index + PSR mask. See docs/BUILD_PROMPTS.md.

Coarsen dem_aoi to compute_res_m (block-mean), sweep n_azimuths horizon angles via a
running-max march to max_horizon_km, derive a [0,1] annual-sunlit-fraction PROXY using the
pole's max solar elevation, then resample index + PSR back to the 5 m grid.

METHOD IS A PROXY: illumination is derived from topography (horizon vs Sun-elevation cap),
not a modelled Sun ephemeris. Sun-elevation assumption: max ~1.53 deg at the lunar S pole.
"""
from __future__ import annotations

import time

import numpy as np

import lunar_ice  # noqa: F401
from lunar_ice import illumination, io_utils


def main() -> None:
    import rasterio
    from rasterio.transform import rowcol
    from scipy.ndimage import zoom

    cfg = io_utils.load_config()
    ic = cfg["illumination"]
    interim = io_utils.resolve_path(cfg, cfg["outputs"]["interim"])
    px = float(cfg["crs"]["pixel_size_m"])
    res = float(ic["compute_res_m"])
    factor = int(round(res / px))
    sun = float(ic["sun_elev_max_deg"])

    with rasterio.open(interim / "dem_aoi.tif") as ds:
        dem = ds.read(1)
        profile = ds.profile.copy()
    H, W = dem.shape

    dem_coarse = illumination.block_mean(dem, factor)
    print(f"coarsen: {dem.shape} @ {px}m -> {dem_coarse.shape} @ {res}m (factor {factor})")
    print(f"sweep: n_azimuths={ic['n_azimuths']}  max_horizon={ic['max_horizon_km']}km  "
          f"sun_elev_max={sun} deg (S-pole assumption)")

    t0 = time.time()
    index_c = illumination.illumination_index(
        dem_coarse, res, int(ic["n_azimuths"]),
        float(ic["max_horizon_km"]) * 1000.0, sun)
    print(f"horizon sweep done in {time.time()-t0:.1f}s")

    # Resample back to the 5 m grid (bilinear for index, nearest for the binary mask).
    index = zoom(index_c, factor, order=1)[:H, :W].astype(np.float32)
    psr = illumination.psr_mask(index_c)
    psr = zoom(psr, factor, order=0)[:H, :W].astype(np.uint8)

    # Save.
    io_utils.write_raster(interim / "illumination_index.tif", index, profile)
    psr_profile = profile.copy()
    psr_profile.update(nodata=None)
    io_utils.write_raster(interim / "psr_mask.tif", psr, psr_profile)
    print(f"\nsaved: {interim/'illumination_index.tif'}")
    print(f"saved: {interim/'psr_mask.tif'}")

    # ---- Sanity ----
    psr_pct = 100.0 * psr.mean()
    print(f"\nAOI in PSR (index ~ 0): {psr_pct:.2f}%")
    print(f"illumination_index: min={index.min():.3f} median={np.median(index):.3f} "
          f"mean={index.mean():.3f} max={index.max():.3f}")

    # F2 crater floor should be PSR (index ~0); rims/highs should be lit (index >0).
    fx, fy = io_utils.lonlat_to_xy(cfg["target"]["lon"], cfg["target"]["lat"], cfg)
    fr, fc = rowcol(profile["transform"], fx, fy)
    floor_val = float(index[fr, fc])
    lo_thr, hi_thr = np.percentile(dem, [10, 90])
    floor_idx = index[dem <= lo_thr]
    high_idx = index[dem >= hi_thr]
    print(f"\nF2 centre (row,col)=({fr},{fc})  elev={dem[fr,fc]:.1f}m  "
          f"illumination_index={floor_val:.3f}  PSR? {bool(psr[fr,fc])}")
    print(f"lowest-elev decile (crater floors): mean index={floor_idx.mean():.3f}  "
          f"PSR frac={100*(floor_idx<=1e-6).mean():.1f}%")
    print(f"highest-elev decile (rims/highs)  : mean index={high_idx.mean():.3f}  "
          f"PSR frac={100*(high_idx<=1e-6).mean():.1f}%")
    ok = floor_val < 0.05 and high_idx.mean() > floor_idx.mean()
    print(f"\nSANITY: F2 floor dark & highs lit? -> {'CONFIRMED' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
