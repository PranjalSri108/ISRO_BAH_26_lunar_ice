"""Runner: 06_dfsar_detect — DFSAR subsurface-ice detection over the F2 AOI.

Window-reads a +/-15 km AOI around F2 from the Chandrayaan-2 DFSAR L3C derived south-pole-east
mosaic (CPR + ODD/EVN/HLX/VOL decomposition), applies the CPR>1 + volume-dominant ice
criterion, validates against the ICY_CRATERS_SP catalogue, and estimates ice volume /
water-equivalent mass. Writes rasters, figures, stats.json and volume.csv to outputs/dfsar/.

This is the derived-mosaic CPR + volume-scattering criterion — the m-chi / Yamaguchi analog of
CPR + DOP (volume scattering => subsurface ice) — assuming ~5 m radar penetration depth.
Ice detection is a hand-off product, kept out of the LOLA terrain/landing module (see CLAUDE.md).
"""
from __future__ import annotations

import csv
import json

import numpy as np

import lunar_ice  # noqa: F401  (ensures package import works)
from lunar_ice import dfsar, io_utils


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = io_utils.load_config()
    d = cfg["dfsar"]
    px = float(d["pixel_size_m"])
    interim = io_utils.resolve_path(cfg, cfg["outputs"]["interim"])
    outdir = io_utils.resolve_path(cfg, d["out"])
    outdir.mkdir(parents=True, exist_ok=True)

    # --- (1) AOI window-read of every layer; assert F2 inside; save intermediates -----------
    bounds = dfsar.aoi_bounds(cfg)
    print(f"=== DFSAR ice detection — F2 AOI ===")
    print(f"F2 X={d['f2_xy'][0]} Y={d['f2_xy'][1]} m  (col,row)={tuple(d['f2_pixel'])}")
    print(f"AOI +/-{d['half_width_m']/1000:.0f} km  bounds (m) "
          f"L={bounds[0]:.0f} B={bounds[1]:.0f} R={bounds[2]:.0f} T={bounds[3]:.0f}")

    bands = dfsar.read_aoi(cfg)
    profile = bands["profile"]
    h, w = profile["height"], profile["width"]
    r0, c0 = bands["f2_rowcol"]
    f2_inside = 0 <= r0 < h and 0 <= c0 < w
    print(f"AOI shape {h} x {w} px   F2 at AOI (row,col)=({r0},{c0})  inside? {f2_inside}")
    assert f2_inside, "F2 is NOT inside the AOI window — check f2_xy / half_width_m in config."

    # Save *_aoi.tif intermediates for every layer (float32, CRS/transform preserved).
    for key in ("cpr", "vol", "odd", "evn", "hlx", "icy"):
        io_utils.write_raster(interim / f"dfsar_{key}_aoi.tif",
                              bands[key].astype(np.float32), profile)

    # --- (2) Ice criterion: CPR>1 AND volume-scattering dominant ----------------------------
    res = dfsar.detect_ice(bands, cfg)
    n_valid = int(res.valid.sum())
    print(f"\nValid mosaic pixels: {n_valid:,} / {h*w:,} ({100*n_valid/(h*w):.1f}%)")
    print(f"CPR>{res.meta['cpr_thresh']:.0f}            : {int((res.cpr_high).sum()):,} px")
    print(f"Volume-dominant (primary): {int(res.vol_dominant.sum()):,} px "
          f"(VOL > ODD+EVN+HLX)")
    n_ice = int(res.ice_mask.sum())
    n_ice_alt = int(res.ice_mask_alt.sum())
    print(f"ICE (CPR>1 & vol-dominant): {n_ice:,} px  (primary)")
    print(f"ICE alt (CPR>1 & VOL>p{int(res.meta['vol_pctile'])}={res.vol_p90:.3e}): {n_ice_alt:,} px")

    # F2 interior PROXY disk + fraction flagged.
    interior = dfsar.f2_interior_mask(profile, (r0, c0),
                                      float(d["f2_interior_radius_m"]), px)
    interior_valid = interior & res.valid
    n_int = int(interior_valid.sum())
    n_int_ice = int((interior_valid & (res.ice_mask > 0)).sum())
    int_ice_pct = 100.0 * n_int_ice / n_int if n_int else 0.0

    # --- (3) Validation against ICY_CRATERS_SP catalogue ------------------------------------
    val = dfsar.validate(res.ice_mask, bands["icy"], res.valid)

    # --- (4) Volume / water-equivalent mass --------------------------------------------------
    vol = dfsar.estimate_volume(n_ice, cfg)
    f2_ice_area_km2 = n_ice * px * px / 1e6

    # --- write rasters ----------------------------------------------------------------------
    io_utils.write_raster(outdir / "cpr_aoi.tif", res.cpr.astype(np.float32), profile)
    io_utils.write_raster(outdir / "vol_aoi.tif", bands["vol"].astype(np.float32), profile)
    io_utils.write_raster(outdir / "ice_mask.tif", res.ice_mask, profile)
    io_utils.write_raster(outdir / "ice_prob.tif", res.ice_prob, profile)

    # --- figure: CPR vs volume-fraction scatter with threshold lines ------------------------
    _scatter_fig(res, outdir / "cpr_vol_scatter.png", plt)

    # --- figure: detected mask vs catalogue over F2 -----------------------------------------
    _overlay_fig(res, bands["icy"], (r0, c0), interior, outdir / "validation_overlay.png", plt)

    # --- stats.json --------------------------------------------------------------------------
    stats = {
        "aoi": {"bounds_m": list(bounds), "shape": [h, w], "pixel_size_m": px,
                "f2_xy_m": list(d["f2_xy"]), "f2_aoi_rowcol": [r0, c0]},
        "criterion": "CPR > 1 AND volume scattering dominant (VOL > ODD+EVN+HLX); "
                     "m-chi/Yamaguchi analog of CPR+DOP, ~5 m penetration assumed",
        "thresholds": res.meta,
        "counts": {
            "valid_px": n_valid, "total_px": h * w,
            "cpr_high_px": int(res.cpr_high.sum()),
            "vol_dominant_px": int(res.vol_dominant.sum()),
            "ice_px_primary": n_ice, "ice_px_alt": n_ice_alt,
            "vol_p90": res.vol_p90,
        },
        "f2_interior": {
            "radius_m": float(d["f2_interior_radius_m"]),
            "valid_px": n_int, "ice_px": n_int_ice, "ice_pct": int_ice_pct,
            "note": "interior is a PROXY disk (no published F2 rim polygon in these data)",
        },
        "ice_area_km2": f2_ice_area_km2,
        "validation": val,
        "validation_note": "ICY_CRATERS_SP nearest-neighbour aligned to the CPR AOI grid; "
                           "recovery is NaN/0 if the catalogue flags no ice inside this AOI",
        "volume": vol,
        "assumptions": [
            "no optical data: detection is radar CPR + decomposition only",
            "VOL/ODD/EVN/HLX are linear scattered-power layers; the vol-dominant test is "
            "scale-invariant",
            "ice_fraction = porosity (pore space fully ice-filled) — an upper-bound assumption",
            "5 m radar penetration depth assumed for the ice column",
        ],
    }
    with open(outdir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    # --- volume.csv --------------------------------------------------------------------------
    with open(outdir / "volume.csv", "w", newline="") as fh:
        cols = ["level", "porosity", "ice_fraction", "ice_area_km2", "depth_m",
                "ice_volume_m3", "water_mass_kg", "water_mass_Mt",
                "eps_birchak", "eps_maxwell_garnett"]
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        for level, row in vol["levels"].items():
            wtr.writerow({"level": level, **{k: row[k] for k in cols[1:]}})

    # --- console summary (the headline numbers) ---------------------------------------------
    lo = vol["levels"]["low"]["water_mass_Mt"]
    ce = vol["levels"]["central"]["water_mass_Mt"]
    hi = vol["levels"]["high"]["water_mass_Mt"]
    print("\n================= F2 DFSAR ICE — SUMMARY =================")
    print(f"F2 ice area               : {f2_ice_area_km2:.3f} km^2  ({n_ice:,} px)")
    print(f"% of F2 interior flagged  : {int_ice_pct:.2f}%  "
          f"({n_int_ice}/{n_int} valid px in {d['f2_interior_radius_m']/1000:.0f} km disk)")
    if val["catalogue_px_in_aoi"] == 0:
        print(f"Overlap with catalogue    : N/A — ICY_CRATERS_SP flags 0 ice px in this AOI")
    else:
        print(f"Overlap with catalogue    : agreement {val['agreement_pct']:.1f}% of detected, "
              f"recovery {val['recovery_pct']:.1f}% of catalogue "
              f"({val['true_positive_px']}/{val['catalogue_px_in_aoi']} px)")
    print(f"Water-equiv mass range    : {lo:.3f} / {ce:.3f} / {hi:.3f} Mt  (low/central/high)")
    print(f"  ice volume (central)    : {vol['levels']['central']['ice_volume_m3']:.3e} m^3 "
          f"@ {d['penetration_depth_m']} m depth, phi={d['porosity']['central']}")
    print("Method: derived-mosaic CPR + volume-scattering criterion "
          "(m-chi/Yamaguchi analog of CPR+DOP; volume scattering = subsurface ice), "
          "5 m penetration assumed.")
    print(f"Outputs -> {outdir}")


def _scatter_fig(res, path, plt) -> None:
    """CPR (x) vs volume-fraction (y) for valid pixels, with the two decision thresholds."""
    v = res.valid
    cpr = res.cpr[v]
    vf = res.vol_frac[v]
    # subsample for a legible scatter if huge
    if cpr.size > 200_000:
        idx = np.random.default_rng(0).choice(cpr.size, 200_000, replace=False)
        cpr, vf = cpr[idx], vf[idx]
    ice = (cpr > res.meta["cpr_thresh"]) & (vf > 0.5)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cpr[~ice], vf[~ice], s=2, c="0.6", alpha=0.3, label="other")
    ax.scatter(cpr[ice], vf[ice], s=3, c="tab:cyan", alpha=0.6, label="ICE (both hold)")
    ax.axvline(res.meta["cpr_thresh"], color="tab:red", ls="--", lw=1.4,
               label=f"CPR = {res.meta['cpr_thresh']:.0f}")
    ax.axhline(0.5, color="tab:blue", ls="--", lw=1.4, label="vol-fraction = 0.5")
    ax.set_xlabel("CPR")
    ax.set_ylabel("volume fraction  VOL / (ODD+EVN+HLX+VOL)")
    ax.set_title("DFSAR ice criterion over F2 AOI\nCPR > 1  AND  volume scattering dominant")
    ax.set_xlim(0, max(2.2, float(np.nanpercentile(cpr, 99.5))))
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", markerscale=3, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _overlay_fig(res, icy, f2_rowcol, interior, path, plt) -> None:
    """Detected ice (cyan) vs ICY catalogue (red) over the CPR backdrop, F2 marked."""
    from matplotlib.lines import Line2D

    r0, c0 = f2_rowcol
    bg = np.where(res.valid, res.cpr, np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.imshow(bg, cmap="gray", origin="upper",
              vmin=0, vmax=float(np.nanpercentile(bg, 99)))
    # catalogue (red) then detected (cyan) as transparent overlays
    cat = np.ma.masked_where(~(icy > 0.5), np.ones_like(icy))
    mine = np.ma.masked_where(res.ice_mask == 0, np.ones_like(icy))
    ax.imshow(cat, cmap=_solid("red"), origin="upper", alpha=0.45)
    ax.imshow(mine, cmap=_solid("cyan"), origin="upper", alpha=0.7)
    # F2 + interior disk outline
    ax.scatter([c0], [r0], s=160, facecolors="none", edgecolors="yellow", linewidths=2)
    ax.annotate("F2", (c0, r0), color="yellow", fontsize=12,
                xytext=(8, 8), textcoords="offset points")
    ax.contour(interior, levels=[0.5], colors="yellow", linewidths=1.0, linestyles=":")
    ax.set_title("DFSAR detected ice vs ICY_CRATERS catalogue — F2 AOI\n"
                 "(CPR backdrop; yellow dotted = F2 interior proxy)")
    ax.set_xlabel("col (px)")
    ax.set_ylabel("row (px)")
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor="cyan",
                      markersize=10, label="detected ice"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
                      markersize=10, label="catalogue ice")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _solid(color):
    """A 1-colour colormap for boolean overlays."""
    from matplotlib.colors import ListedColormap
    return ListedColormap([color])


if __name__ == "__main__":
    main()
