"""Runner: 05_figures — Publication-quality hero map + cross-section, and hand-off bundle.

FIGURE 1  outputs/hero_landing_map.png : AOI hillshade + balanced suitability (green=best) +
          PSR (cool blue) + F2 outline + per-profile sites + distance lines + callout +
          scale bar + north arrow.
FIGURE 2  outputs/cross_section_F2.png : DEM profile recommended-site -> F2 centre with the
          bench / rim / PSR-floor zones annotated and slope-along-profile on a twin axis.
Then the outputs/ hand-off bundle (rasters + geojsons + manifest.json, shared grid/CRS).
See docs/BUILD_PROMPTS.md.
"""
from __future__ import annotations

import numpy as np

import lunar_ice  # noqa: F401
from lunar_ice import io_utils, viz

# Colourblind-safe (Okabe-Ito) palette for the per-profile markers.
PROFILE_STYLE = {
    "safest":      {"color": "#0072B2", "marker": "o", "label": "safest"},
    "closest_ice": {"color": "#D55E00", "marker": "s", "label": "closest_ice"},
    "best_lit":    {"color": "#E69F00", "marker": "^", "label": "best_lit"},
    "balanced":    {"color": "#009E73", "marker": "*", "label": "balanced (recommended)"},
}


def _read(interim, name):
    import rasterio
    with rasterio.open(interim / name) as ds:
        return ds.read(1), ds.profile.copy()


def _north_vec(cfg, lon, lat):
    """Unit (dcol, drow) of geographic north (increasing latitude) in image pixels."""
    x0, y0 = io_utils.lonlat_to_xy(lon, lat, cfg)
    x1, y1 = io_utils.lonlat_to_xy(lon, lat + 0.25, cfg)  # a step toward the equator = N
    dcol, drow = (x1 - x0), -(y1 - y0)  # +x = +col, +y = -row (north-up raster)
    n = np.hypot(dcol, drow)
    return dcol / n, drow / n


def _scale_bar(ax, px_size_m, length_m=5000, loc=(250, 5650)):
    import matplotlib.patches as mp
    x0, y0 = loc
    length_px = length_m / px_size_m
    ax.add_patch(mp.Rectangle((x0, y0), length_px, 70, facecolor="k", edgecolor="k"))
    ax.add_patch(mp.Rectangle((x0, y0), length_px / 2, 70, facecolor="w", edgecolor="k"))
    ax.text(x0 + length_px / 2, y0 - 60, f"{length_m/1000:.0f} km", ha="center",
            va="bottom", fontsize=15, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))


def _north_arrow(ax, cfg, lon, lat, anchor=(5450, 1250), length=520):
    dcol, drow = _north_vec(cfg, lon, lat)
    hx, hy = anchor[0] + length * dcol, anchor[1] + length * drow
    ax.annotate("", xy=(hx, hy), xytext=anchor,
                arrowprops=dict(facecolor="k", edgecolor="k", width=5, headwidth=18))
    ax.text(hx + 30 * dcol, hy + 30 * drow, "N", fontsize=18, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="k"))


def build_hero(cfg, interim, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    import geopandas as gpd
    from rasterio.transform import rowcol

    dem, profile = _read(interim, "dem_aoi.tif")
    bal, _ = _read(interim, "suitability_balanced.tif")
    psr, _ = _read(interim, "psr_mask.tif")
    transform = profile["transform"]
    px = float(cfg["crs"]["pixel_size_m"])

    cand = gpd.read_file(out_dir / "landing_candidates.geojson")
    f2g = gpd.read_file(out_dir / "target_crater.geojson").iloc[0]
    fx, fy = io_utils.lonlat_to_xy(cfg["target"]["lon"], cfg["target"]["lat"], cfg)
    f2r, f2c = rowcol(transform, fx, fy)
    rim_px = float(f2g["rim_radius_m"]) / px

    sites = {}
    for _, r in cand.iterrows():
        x, y = io_utils.lonlat_to_xy(r.geometry.x, r.geometry.y, cfg)
        rr, cc = rowcol(transform, x, y)
        sites[r["profile"]] = {**r.drop("geometry").to_dict(), "row": rr, "col": cc}

    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(15, 13))
    ax.imshow(viz.hillshade(dem, px), cmap="gray", origin="upper")

    # Suitability green(best)->red(worst), semi-transparent, masked outside go-areas.
    suit = np.ma.masked_where(bal <= 0, bal)
    im = ax.imshow(suit, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.0455, pad=0.02)
    cb.set_label("Balanced landing suitability (green = best)", fontsize=15)

    # PSR cold-trap context, cool blue.
    psr_cmap = ListedColormap([(0.10, 0.45, 0.95, 1.0)])
    ax.imshow(np.ma.masked_where(psr == 0, psr), cmap=psr_cmap, alpha=0.45)

    # F2 outline (white) + centre.
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(f2c + rim_px * np.cos(th), f2r + rim_px * np.sin(th), color="white", lw=2.5)
    ax.plot(f2c, f2r, "x", color="white", ms=12, mew=3)
    ax.annotate("F2 (Faustini)", (f2c, f2r), color="white", fontsize=16, fontweight="bold",
                xytext=(10, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="white", alpha=0.55))

    # Sites + distance lines to F2.
    for pname, s in sites.items():
        st = PROFILE_STYLE[pname]
        ax.plot([s["col"], f2c], [s["row"], f2r], "--", color=st["color"], lw=1.8,
                alpha=0.85)
        big = pname == "balanced"
        ax.scatter([s["col"]], [s["row"]], s=420 if big else 230, marker=st["marker"],
                   facecolors=st["color"], edgecolors="white", linewidths=2.2, zorder=5)
        mx, my = (s["col"] + f2c) / 2, (s["row"] + f2r) / 2
        ax.annotate(f"{s['dist_to_F2_km']:.1f} km", (mx, my), color="black", fontsize=13,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=st["color"], alpha=0.9))

    # Legend (markers).
    handles = [Line2D([0], [0], marker=st["marker"], color="none",
                      markerfacecolor=st["color"], markeredgecolor="white",
                      markersize=16, label=st["label"]) for st in PROFILE_STYLE.values()]
    handles.append(Line2D([0], [0], marker="s", color="none",
                          markerfacecolor=(0.10, 0.45, 0.95), markersize=14,
                          label="PSR (permanent shadow)"))
    leg = ax.legend(handles=handles, loc="upper left", fontsize=13, framealpha=0.9,
                    title="Top site per objective", title_fontsize=13)
    leg.get_frame().set_edgecolor("k")

    # Metric callout for the recommended (balanced) site.
    b = sites["balanced"]
    txt = ("RECOMMENDED — balanced\n"
           f"slope:  {b['slope_mean']:.1f} deg mean / {b['slope_max']:.1f} deg max\n"
           f"roughness:  {b['roughness_pct']:.0f}th pct\n"
           f"dist to F2:  {b['dist_to_F2_km']:.2f} km\n"
           f"illumination:  {b['illum_index']:.2f}  (~{b['lit_pct']:.0f}% lit)")
    ax.text(0.985, 0.015, txt, transform=ax.transAxes, fontsize=14, ha="right", va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFFDE7", ec="#009E73", lw=2))

    _scale_bar(ax, px)
    _north_arrow(ax, cfg, cfg["target"]["lon"], cfg["target"]["lat"])

    ax.set_title("Candidate landing sites near doubly shadowed crater F2 (Faustini)",
                 fontsize=21, fontweight="bold", pad=14)
    ax.set_xlabel("AOI easting (px, 5 m)", fontsize=14)
    ax.set_ylabel("AOI northing (px, 5 m)", fontsize=14)
    fig.tight_layout()
    out = out_dir / "hero_landing_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out, sites


def build_cross_section(cfg, interim, out_dir, sites):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.transform import rowcol

    dem, profile = _read(interim, "dem_aoi.tif")
    slope, _ = _read(interim, "slope_aoi.tif")
    psr, _ = _read(interim, "psr_mask.tif")
    transform = profile["transform"]
    px = float(cfg["crs"]["pixel_size_m"])
    H, W = dem.shape

    fx, fy = io_utils.lonlat_to_xy(cfg["target"]["lon"], cfg["target"]["lat"], cfg)
    f2r, f2c = rowcol(transform, fx, fy)
    b = sites["balanced"]
    sr, sc = b["row"], b["col"]

    # Sample the straight line site -> F2 centre at ~5 m spacing.
    n = max(2, int(round(np.hypot(f2r - sr, f2c - sc))))
    rs = np.linspace(sr, f2r, n).round().astype(int).clip(0, H - 1)
    cs = np.linspace(sc, f2c, n).round().astype(int).clip(0, W - 1)
    dist_km = np.hypot((cs - sc) * px, (rs - sr) * px) / 1000.0
    elev = dem[rs, cs]
    slp = slope[rs, cs]
    psr_line = psr[rs, cs]

    # Zones. PSR floor = the contiguous shadow run reaching F2 centre.
    psr_idx = np.where(psr_line == 1)[0]
    floor_start = int(psr_idx[0]) if psr_idx.size else n - 1
    rim_idx = int(np.argmax(elev[:floor_start + 1])) if floor_start > 0 else 0
    # Bench = flat run from the site up to where slope first leaves the gentle limit.
    slope_max = float(cfg["constraints"]["slope_max_deg"])
    leave = np.where(slp[:rim_idx + 1] > slope_max)[0]
    bench_end = int(leave[0]) if leave.size else rim_idx

    elev_floor = float(np.min(elev[floor_start:]))
    descent = float(elev[0] - elev_floor)
    rim_height = float(elev[rim_idx] - np.median(elev[:max(1, bench_end) + 1]))
    rim_slope = (float(np.max(slp[bench_end:floor_start + 1]))
                 if floor_start > bench_end else float(slp[rim_idx]))

    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.axvspan(dist_km[0], dist_km[bench_end], color="#A6D854", alpha=0.30,
               label="landing bench (flat)")
    ax.axvspan(dist_km[bench_end], dist_km[floor_start], color="#FC8D62", alpha=0.25,
               label="crater rim / wall")
    ax.axvspan(dist_km[floor_start], dist_km[-1], color="#3B6FB5", alpha=0.28,
               label="PSR floor (ice target)")

    ax.plot(dist_km, elev, color="black", lw=3, zorder=4, label="topographic profile")
    ax.plot(dist_km[0], elev[0], marker="*", ms=26, color="#009E73",
            markeredgecolor="white", mew=1.5, zorder=6)
    ax.annotate("landing site\n(balanced)", (dist_km[0], elev[0]), fontsize=14,
                xytext=(12, 18), textcoords="offset points", fontweight="bold")
    ax.plot(dist_km[rim_idx], elev[rim_idx], marker="v", ms=16, color="#FC8D62",
            markeredgecolor="k", zorder=6)
    ax.annotate(f"rim crest\n+{rim_height:.0f} m, slope {rim_slope:.0f} deg",
                (dist_km[rim_idx], elev[rim_idx]), fontsize=13, xytext=(8, 10),
                textcoords="offset points")
    ax.axvline(dist_km[floor_start], color="#1B3A6B", ls=":", lw=2)
    ax.annotate("PSR begins\n(psr_mask = 1)", (dist_km[floor_start], elev_floor),
                fontsize=13, xytext=(8, 30), textcoords="offset points", color="#1B3A6B")

    ax2 = ax.twinx()
    ax2.plot(dist_km, slp, color="#7B3294", lw=1.4, alpha=0.7)
    ax2.axhline(slope_max, color="#7B3294", ls="--", lw=1, alpha=0.6)
    ax2.set_ylabel("slope along profile (deg)", color="#7B3294", fontsize=14)
    ax2.tick_params(axis="y", colors="#7B3294")
    ax2.set_ylim(0, max(35, float(slp.max()) * 1.1))

    callout = (f"landing slope: {b['slope_mean']:.1f} deg mean, {b['slope_max']:.1f} deg max\n"
               f"traverse to F2: {dist_km[-1]:.2f} km\n"
               f"total descent to floor: {descent:.0f} m")
    ax.text(0.015, 0.04, callout, transform=ax.transAxes, fontsize=14, va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFFDE7", ec="#009E73", lw=2))

    ax.set_xlabel("traverse distance from landing site toward F2 centre (km)", fontsize=15)
    ax.set_ylabel("elevation (m)", fontsize=15)
    ax.set_title("Rover approach geometry: landing bench -> F2 rim -> "
                 "permanently shadowed floor", fontsize=19, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.92)
    fig.tight_layout()
    out = out_dir / "cross_section_F2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def write_bundle(cfg, interim, out_dir):
    px = float(cfg["crs"]["pixel_size_m"])
    layer_specs = [
        {"name": "suitability.tif", "src": interim / "suitability_balanced.tif",
         "description": "Balanced multi-criteria landing suitability", "units": "[0,1]"},
        {"name": "hazard.tif", "src": interim / "hazard.tif",
         "description": "Hard no-go mask (steep/rough/high-relief/crater/too-far)",
         "units": "{0,1}"},
        {"name": "slope.tif", "src": interim / "slope_aoi.tif",
         "description": "Surface slope (LOLA LDSM)", "units": "deg"},
        {"name": "roughness.tif", "src": interim / "roughness.tif",
         "description": "Local slope std (BOULDER PROXY, no optical)", "units": "deg"},
        {"name": "illumination_index.tif", "src": interim / "illumination_index.tif",
         "description": "Horizon-based annual-sunlit-fraction PROXY", "units": "[0,1]"},
        {"name": "psr_mask.tif", "src": interim / "psr_mask.tif",
         "description": "Permanently-shadowed-region proxy (index ~ 0)", "units": "{0,1}"},
    ]
    vector_specs = [
        {"name": "landing_candidates.geojson", "description": "Ranked landing points + metrics"},
        {"name": "landing_site_polygon.geojson", "description": "75 m safety ellipse, best site"},
        {"name": "target_crater.geojson", "description": "F2 (Faustini) crater outline"},
    ]
    return viz.write_handoff(layer_specs, vector_specs, out_dir, px)


def main() -> None:
    cfg = io_utils.load_config()
    interim = io_utils.resolve_path(cfg, cfg["outputs"]["interim"])
    out_dir = io_utils.resolve_path(cfg, cfg["outputs"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    hero, sites = build_hero(cfg, interim, out_dir)
    print(f"FIGURE 1 hero         : {hero}")
    xs = build_cross_section(cfg, interim, out_dir, sites)
    print(f"FIGURE 2 cross-section: {xs}")

    manifest = write_bundle(cfg, interim, out_dir)
    print(f"\nhand-off bundle (shared grid/CRS validated): {len(manifest['rasters'])} rasters "
          f"+ {len(manifest['vectors'])} vectors + manifest.json")
    print(f"  grid: {manifest['rasters'][0]['width']}x{manifest['rasters'][0]['height']} px "
          f"@ {manifest['pixel_size_m']} m, CRS ok")


if __name__ == "__main__":
    main()
