"""Runner: 08_fp_detect — true full-pol CPR + DOP subsurface-ice detection on F2.

Processes the calibrated full-pol complex SLI scene 20200321t082617351 over F2's floor:

  (1) read the 4 complex channels HH/HV/VH/VV (Band1=I, Band2=Q) in azimuth windows,
  (2) HV/VH reciprocity-average; 3x3 boxcar multilook the intensity/covariance terms, then
      form the linear-basis Stokes vector
          S1 = <|HH|^2> + 2<|HV|^2> + <|VV|^2>   (total power)
          S2 = <|HH|^2> - <|VV|^2>
          S3 =  2 Re<HH conj(VV)>,   S4 = -2 Im<HH conj(VV)>,
  (3) CPR = (S1-S4)/(S1+S4),  DOP = sqrt(S2^2+S3^2+S4^2)/S1  (asserted DOP in [0,1], CPR>0),
  (4) ICE = CPR > 1 AND DOP < 0.13   (Sinha et al. 2026, author-confirmed),
  (5) geocode CPR/DOP/validity to the 25 m polar-stereo grid via GCPs from the g_sli geometry
      CSV, crop to the +/-15 km F2 AOI, mask with sri_ma + nodata,
  (6) validate vs ICY_CRATERS_SP (reported honestly), (7) MG+Birchak volume.

True full-pol Stokes CPR + DOP, criterion CPR>1 & DOP<0.13 per Sinha et al. 2026
(author-confirmed), computed on F2's floor; 5 m radar penetration assumed. Low DOP =
depolarised = volume scattering = subsurface ice. Ice detection stays out of the landing
module (CLAUDE.md).
"""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import lunar_ice  # noqa: F401
from lunar_ice import dfsar, io_utils


def _channel_paths(cfg):
    f = cfg["fp"]
    base = io_utils.resolve_path(cfg, f["scene_dir"])
    return {ch: base / (f["stem"] + suf) for ch, suf in f["channels"].items()}


def _read_complex(ds, r0, n):
    """Read azimuth window [r0:r0+n] of a 2-band (I,Q) SLI as complex64."""
    from rasterio.windows import Window
    w = Window(0, r0, ds.width, n)
    return ds.read(1, window=w).astype(np.float32) + 1j * ds.read(2, window=w).astype(np.float32)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.transform import rowcol
    from pyproj import CRS, Transformer
    from scipy.ndimage import uniform_filter

    cfg = io_utils.load_config()
    f, d = cfg["fp"], cfg["dfsar"]
    look = int(f["multilook"])
    core = int(f["az_window"])
    outdir = io_utils.resolve_path(cfg, f["out"])
    outdir.mkdir(parents=True, exist_ok=True)

    chan = _channel_paths(cfg)
    gcp_ds = rasterio.open(chan[f["gcp_channel"]])
    H, W = gcp_ds.height, gcp_ds.width
    _, gcp_crs = gcp_ds.gcps                          # geographic CRS for the GCPs (GCS_MOON)
    gcp_ds.close()
    with rasterio.open(io_utils.resolve_path(cfg, f["scene_dir"]) / f["sri_mask"]) as ds:
        sri_crs = ds.crs

    # --- GCPs from the g_sli geometry CSV ---------------------------------------------------
    grows, gcols, glon, glat, (n_az, n_rg) = dfsar.read_gsli_gcps(
        io_utils.resolve_path(cfg, f["gsli_csv"]), H, W)
    print(f"=== Full-pol CPR+DOP ice detection — F2 ===")
    print(f"g_sli geometry grid: {n_az} az x {n_rg} rg = {grows.size:,} GCP nodes")

    # AOI azimuth band: CSV nodes whose projected position falls in the F2 AOI box.
    fx, fy = d["f2_xy"]; hw = float(d["half_width_m"])
    tf = Transformer.from_crs(CRS.from_wkt(gcp_crs.to_wkt()),
                              CRS.from_wkt(sri_crs.to_wkt()), always_xy=True)
    gX, gY = tf.transform(glon, glat)
    gX = np.asarray(gX); gY = np.asarray(gY)
    inb = (np.abs(gX - fx) <= hw) & (np.abs(gY - fy) <= hw)
    if not inb.any():
        raise SystemExit("No g_sli GCP nodes project inside the F2 AOI.")
    pad = int(f["az_pad_rows"])
    az0 = max(0, int(np.floor(grows[inb].min())) - pad)
    az1 = min(H, int(np.ceil(grows[inb].max())) + pad)
    az0 -= az0 % look                                # align subsample phase to az0
    print(f"Scene rows {H:,}; processing azimuth band [{az0:,}:{az1:,}] "
          f"({az1-az0:,} rows, {(az1-az0)//core+1} windows)")

    # --- (1)-(2) windowed read -> multilooked linear-basis Stokes terms ---------------------
    open_ch = {ch: rasterio.open(p) for ch, p in chan.items()}
    acc = {k: [] for k in ("ihh", "ihv", "ivv", "re", "im", "vf")}
    for s in range(az0, az1, core):
        core_end = min(s + core, az1)
        lo, hi = max(0, s - look), min(H, core_end + look)
        hh = _read_complex(open_ch["hh"], lo, hi - lo)
        hv = _read_complex(open_ch["hv"], lo, hi - lo)
        vh = _read_complex(open_ch["vh"], lo, hi - lo)
        vv = _read_complex(open_ch["vv"], lo, hi - lo)
        ihh, ihv, ivv, re, im = dfsar.linear_stokes_terms(hh, hv, vh, vv)
        good = ((np.abs(hh) > 0) & (np.abs(vv) > 0)).astype(np.float64)
        a, b = s - lo, core_end - lo
        sub = (slice(a + look // 2, b, look), slice(look // 2, None, look))  # core, multilooked
        for key, arr in (("ihh", ihh), ("ihv", ihv), ("ivv", ivv),
                         ("re", re), ("im", im), ("vf", good)):
            acc[key].append(uniform_filter(arr, look, mode="nearest")[sub])
    for ds in open_ch.values():
        ds.close()
    M = {k: np.vstack(v) for k, v in acc.items()}
    H_ml, W_ml = M["ihh"].shape

    # --- (3) Stokes -> CPR / DOP, with sanity asserts ---------------------------------------
    S1, S2, S3, S4 = dfsar.stokes_from_terms(M["ihh"], M["ihv"], M["ivv"], M["re"], M["im"])
    cpr_sl, dop_sl = dfsar.stokes_to_cpr_dop(S1, S2, S3, S4)
    valid_sl = (M["vf"] > 0.5) & np.isfinite(cpr_sl) & np.isfinite(dop_sl)
    dv, cv = dop_sl[valid_sl], cpr_sl[valid_sl]
    dmin, dmax, cmin = float(dv.min()), float(dv.max()), float(cv.min())
    print(f"Multilooked slant grid: {H_ml} x {W_ml} (valid {100*valid_sl.mean():.1f}%); "
          f"DOP[{dmin:.4f},{dmax:.4f}] CPRmin {cmin:.4f}")
    assert -1e-3 <= dmin and dmax <= 1 + 1e-3, f"DOP outside [0,1]: {dmin}..{dmax}"
    assert cmin >= -1e-3, f"CPR went negative: {cmin}"
    dop_sl = np.clip(dop_sl, 0.0, 1.0)               # clip tiny numerical excursions
    cpr_sl = np.maximum(cpr_sl, 0.0)
    valid_f = valid_sl.astype(np.float32)

    # --- (5) geocode via g_sli GCPs (scaled to the multilooked grid, TPS-subsampled) --------
    az_idx = (grows / max((H - 1) / max(n_az - 1, 1), 1e-9)).round().astype(int)
    band_az = np.unique(az_idx[(grows >= az0) & (grows <= az1)])
    stride = max(1, int(np.ceil(band_az.size / int(f["max_gcp_az_lines"]))))
    keep_lines = set(band_az[::stride].tolist())
    scaled = []
    for r, c, lo_, la_, ai in zip(grows, gcols, glon, glat, az_idx):
        ml_row, ml_col = (r - az0) / look, c / look
        if ai in keep_lines and -2 <= ml_row <= H_ml + 2 and -2 <= ml_col <= W_ml + 2:
            scaled.append(GroundControlPoint(row=ml_row, col=ml_col, x=lo_, y=la_, z=0.0))
    print(f"GCPs used for geocoding (TPS): {len(scaled)} "
          f"({len(keep_lines)} az lines x {n_rg} rg)")

    tmpdir = Path(tempfile.mkdtemp(prefix="fp_geo_"))
    tmp_gcp = tmpdir / "stokes_slant.tif"
    with rasterio.open(tmp_gcp, "w", driver="GTiff", dtype="float32", count=3,
                       height=H_ml, width=W_ml, nodata=np.nan) as dst:
        dst.write(cpr_sl.astype(np.float32), 1)
        dst.write(dop_sl.astype(np.float32), 2)
        dst.write(valid_f, 3)
        dst.gcps = (scaled, gcp_crs)

    # AOI target grid = sri_ma window around F2 (snaps to the 25 m grid; sri_ma aligns exactly)
    aoi_bounds = (fx - hw, fy - hw, fx + hw, fy + hw)
    sri_ma_aoi, aoi_profile = io_utils.window_read(
        io_utils.resolve_path(cfg, f["scene_dir"]) / f["sri_mask"], aoi_bounds)
    t = aoi_profile["transform"]
    L, T = t.c, t.f
    R, B = L + aoi_profile["width"] * t.a, T + aoi_profile["height"] * t.e
    srs_file = tmpdir / "target.wkt"
    srs_file.write_text(sri_crs.to_wkt())
    warp_tif = tmpdir / "stokes_aoi.tif"
    subprocess.run(["gdalwarp", "-q", "-overwrite", "-tps", "-r", "near",
                    "-t_srs", str(srs_file), "-te", str(L), str(B), str(R), str(T),
                    "-tr", "25", "25", "-dstnodata", "nan", str(tmp_gcp), str(warp_tif)],
                   check=True)
    with rasterio.open(warp_tif) as ds:
        CPR, DOP, valid_geo = ds.read(1), ds.read(2), ds.read(3) > 0.5
    aoi_profile = dict(aoi_profile, dtype="float32", nodata=np.nan, count=1)

    # --- (4) mask + ice criterion in the AOI ------------------------------------------------
    valid = valid_geo & (sri_ma_aoi > 0) & np.isfinite(CPR) & np.isfinite(DOP)
    ice_mask, ice_prob = dfsar.fp_ice(CPR, DOP, valid, cfg)
    CPR = np.where(valid, CPR, np.nan).astype(np.float32)
    DOP = np.where(valid, DOP, np.nan).astype(np.float32)

    r0, c0 = rowcol(aoi_profile["transform"], fx, fy)
    interior = dfsar.f2_interior_mask(aoi_profile, (r0, c0),
                                      float(d["f2_interior_radius_m"]), 25.0)
    interior_valid = interior & valid
    n_int = int(interior_valid.sum())
    n_int_ice = int((interior_valid & (ice_mask > 0)).sum())
    int_pct = 100.0 * n_int_ice / n_int if n_int else 0.0

    px = 25.0
    n_ice = int(ice_mask.sum())
    ice_area_km2 = n_ice * px * px / 1e6

    cpr_high = valid & (CPR > float(f["cpr_thresh"]))
    n_013 = int((cpr_high & (DOP < float(f["dop_thresh"]))).sum())
    n_087 = int((cpr_high & (DOP < float(f["dop_team_error"]))).sum())
    area_013, area_087 = n_013 * px * px / 1e6, n_087 * px * px / 1e6
    cpr_med = float(np.nanmedian(CPR[valid])) if valid.any() else float("nan")
    cpr_max = float(np.nanmax(CPR[valid])) if valid.any() else float("nan")
    dop_med = float(np.nanmedian(DOP[valid])) if valid.any() else float("nan")

    # --- (6) validation vs ICY_CRATERS_SP ---------------------------------------------------
    icy = dfsar.align_to_profile(io_utils.resolve_path(cfg, d["icy_mask"]),
                                 aoi_bounds, aoi_profile)
    val = dfsar.validate(ice_mask, icy, valid)

    # --- (7) volume / mass ------------------------------------------------------------------
    vol = dfsar.estimate_volume(n_ice, cfg)

    # --- (8) write rasters + figure + stats + csv -------------------------------------------
    io_utils.write_raster(outdir / "CPR.tif", CPR, aoi_profile)
    io_utils.write_raster(outdir / "DOP.tif", DOP, aoi_profile)
    io_utils.write_raster(outdir / "ice_mask.tif", ice_mask, dict(aoi_profile, nodata=255))
    io_utils.write_raster(outdir / "ice_prob.tif", ice_prob, aoi_profile)
    _scatter(CPR, DOP, valid, f, outdir / "cpr_dop_scatter.png", plt)

    stats = {
        "scene": f["stem"], "method": "true full-pol Stokes CPR + DOP (linear-basis)",
        "criterion": "CPR > 1 AND DOP < 0.13 (Sinha et al. 2026, author-confirmed); "
                     "low DOP = volume scattering = subsurface ice; 5 m penetration assumed",
        "stokes": "S1=|HH|^2+2|HV|^2+|VV|^2; S2=|HH|^2-|VV|^2; S3=2Re(HH conj VV); "
                  "S4=-2Im(HH conj VV); 3x3 boxcar multilook before forming Stokes",
        "aoi": {"bounds_m": [L, B, R, T], "shape": [aoi_profile["height"], aoi_profile["width"]],
                "f2_xy_m": list(d["f2_xy"]), "f2_aoi_rowcol": [int(r0), int(c0)]},
        "processing": {"azimuth_band": [az0, az1], "multilook": look,
                       "slant_grid": [H_ml, W_ml], "n_gcps_used": len(scaled),
                       "gcp_source": "g_sli geometry CSV", "geocode": "GCP TPS warp, near"},
        "sanity": {"dop_min": dmin, "dop_max": dmax, "cpr_min": cmin},
        "valid_px": int(valid.sum()),
        "cpr_median": cpr_med, "cpr_max": cpr_max, "dop_median": dop_med,
        "ice_px": n_ice, "ice_area_km2": ice_area_km2,
        "f2_interior": {"radius_m": float(d["f2_interior_radius_m"]),
                        "valid_px": n_int, "ice_px": n_int_ice, "ice_pct": int_pct,
                        "note": "interior is a PROXY disk (no published F2 rim polygon)"},
        "threshold_contrast": {
            "dop_author_0p13": {"ice_px": n_013, "ice_area_km2": area_013},
            "dop_team_error_0p87": {"ice_px": n_087, "ice_area_km2": area_087},
            "note": "DOP<0.87 (team error) floods in surface scatter; DOP<0.13 isolates ice"},
        "validation": val,
        "validation_note": "ICY_CRATERS_SP nearest-neighbour aligned to AOI; non-overlap "
                           "expected (nearest catalogue ice ~20 km from F2)",
        "volume": vol,
        "masking": "sri_ma > 0 (in-swath valid) AND geocode validity AND finite CPR/DOP",
    }
    with open(outdir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, allow_nan=False, default=lambda o: None)

    with open(outdir / "volume.csv", "w", newline="") as fh:
        cols = ["level", "porosity", "ice_fraction", "ice_area_km2", "depth_m",
                "ice_volume_m3", "water_mass_kg", "water_mass_Mt",
                "eps_birchak", "eps_maxwell_garnett"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for level, row in vol["levels"].items():
            w.writerow({"level": level, **{k: row[k] for k in cols[1:]}})

    lo, ce, hi = (vol["levels"][k]["water_mass_Mt"] for k in ("low", "central", "high"))
    print("\n============== F2 FULL-POL CPR+DOP — SUMMARY ==============")
    print(f"F2 ice area (CPR>1 & DOP<0.13) : {ice_area_km2:.4f} km^2  ({n_ice:,} px)")
    print(f"% of F2 interior flagged       : {int_pct:.2f}%  ({n_int_ice}/{n_int} valid px)")
    print(f"CPR median / max               : {cpr_med:.3f} / {cpr_max:.3f}")
    print(f"DOP median                     : {dop_med:.3f}")
    print(f"Ice area DOP<0.13 (author)     : {area_013:.4f} km^2  ({n_013:,} px)")
    print(f"Ice area DOP<0.87 (team error) : {area_087:.4f} km^2  ({n_087:,} px)  "
          f"-> {area_087/area_013 if area_013 else float('nan'):.0f}x inflation")
    if val["catalogue_px_in_aoi"] == 0:
        print(f"Catalogue overlap              : N/A — 0 ICY px in AOI (expected)")
    else:
        print(f"Catalogue overlap              : agree {val['agreement_pct']:.1f}%, "
              f"recover {val['recovery_pct']:.1f}%")
    print(f"Water-equiv mass (lo/ce/hi)    : {lo:.4f} / {ce:.4f} / {hi:.4f} Mt")
    print("Method: true full-pol Stokes CPR + DOP, criterion CPR>1 & DOP<0.13 "
          "(Sinha et al. 2026, author-confirmed), on F2's floor, 5 m penetration assumed.")
    print(f"Outputs -> {outdir}")


def _scatter(CPR, DOP, valid, f, path, plt):
    """CPR (x) vs DOP (y) density with both DOP threshold lines + CPR=1."""
    c, dp = CPR[valid], DOP[valid]
    m = np.isfinite(c) & np.isfinite(dp)
    c, dp = c[m], dp[m]
    if c.size > 200_000:
        idx = np.random.default_rng(0).choice(c.size, 200_000, replace=False)
        c, dp = c[idx], dp[idx]
    ice = (c > float(f["cpr_thresh"])) & (dp < float(f["dop_thresh"]))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(c[~ice], dp[~ice], s=2, c="0.6", alpha=0.25, label="other")
    ax.scatter(c[ice], dp[ice], s=6, c="tab:cyan", alpha=0.8, label="ICE (CPR>1 & DOP<0.13)")
    ax.axvline(float(f["cpr_thresh"]), color="tab:red", ls="--", lw=1.4, label="CPR = 1")
    ax.axhline(float(f["dop_thresh"]), color="tab:green", ls="--", lw=1.6,
               label="DOP = 0.13 (author)")
    ax.axhline(float(f["dop_team_error"]), color="tab:orange", ls=":", lw=1.6,
               label="DOP = 0.87 (team error)")
    ax.set_xlabel("CPR = (S1 - S4) / (S1 + S4)")
    ax.set_ylabel("DOP = sqrt(S2^2+S3^2+S4^2) / S1")
    ax.set_title("Full-pol Stokes CPR vs DOP over F2 AOI\nice = CPR > 1 AND DOP < 0.13")
    ax.set_xlim(0, max(2.2, float(np.nanpercentile(c, 99.5))))
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", markerscale=2.5, framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
