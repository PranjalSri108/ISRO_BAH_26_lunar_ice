"""Runner: 09_pipeline_figure — one hero figure of the end-to-end F2 pipeline.

Three panels, left -> right, with arrows between them, on REAL Faustini-F2 data:
  (1) Detect   — DFSAR full-pol CPR/DOP ice mask over F2 (outputs/fp_f2/).
  (2) Land     — LOLA AOI hillshade + suitability + the ranked landing site + F2 outline.
  (3) Traverse — the rover route over the detected ice nodes (data/processed/).

Exports outputs/pipeline_hero.png (slide-proportioned, ~13x5, 200 dpi). Read-only over the
pipeline outputs; produces nothing the downstream consumes — this is a presentation artifact.
"""
from __future__ import annotations

import csv

import numpy as np

import lunar_ice  # noqa: F401
from lunar_ice import io_utils, viz


def _ring_to_px(ring_lonlat, dst_crs, transform):
    """F2 crater ring (lon/lat) -> (cols, rows) on a raster's grid."""
    from pyproj import CRS, Transformer
    from rasterio.transform import rowcol
    tf = Transformer.from_crs(io_utils.MOON_GEOGRAPHIC, CRS.from_wkt(dst_crs.to_wkt()),
                              always_xy=True)
    xs, ys = tf.transform([p[0] for p in ring_lonlat], [p[1] for p in ring_lonlat])
    rc = [rowcol(transform, x, y) for x, y in zip(xs, ys)]
    return [c for _, c in rc], [r for r, _ in rc]


def _lonlat_to_px(lon, lat, dst_crs, transform):
    from pyproj import CRS, Transformer
    from rasterio.transform import rowcol
    tf = Transformer.from_crs(io_utils.MOON_GEOGRAPHIC, CRS.from_wkt(dst_crs.to_wkt()),
                              always_xy=True)
    x, y = tf.transform(lon, lat)
    r, c = rowcol(transform, x, y)
    return c, r


def _panel_detect(ax, cfg):
    """(1) CPR backdrop + cyan ice mask + F2 outline."""
    import json
    import rasterio
    from matplotlib.colors import ListedColormap

    out = io_utils.resolve_path(cfg, cfg["fp"]["out"])
    with rasterio.open(out / "CPR.tif") as ds:
        cpr = ds.read(1); crs = ds.crs; transform = ds.transform
    ice = rasterio.open(out / "ice_mask.tif").read(1)

    from scipy.ndimage import binary_dilation

    bg = np.where(np.isfinite(cpr), cpr, np.nan)
    ax.imshow(bg, cmap="gray", origin="upper", vmin=0,
              vmax=float(np.nanpercentile(bg, 97)))
    # Dilate the 25 m ice pixels (0.05% of frame) so the detections read at slide scale.
    ice_vis = binary_dilation(ice == 1, iterations=4)
    ax.imshow(np.ma.masked_where(~ice_vis, np.ones_like(ice)),
              cmap=ListedColormap(["cyan"]), origin="upper", alpha=0.95)

    crater = io_utils.resolve_path(cfg, cfg["outputs"]["out"]) / "target_crater.geojson"
    ring = json.load(open(crater))["features"][0]["geometry"]["coordinates"][0]
    fc, fr = _ring_to_px(ring, crs, transform)
    ax.plot(fc, fr, color="yellow", lw=1.6)
    cx, cy = np.mean(fc), np.mean(fr)
    ax.annotate("F2", (cx, cy), color="yellow", fontsize=11, weight="bold",
                xytext=(8, 8), textcoords="offset points")
    ax.set_title("DFSAR ice detection: 0.47 km$^2$, CPR>1 & DOP<0.13", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])


def _panel_land(ax, cfg):
    """(2) LOLA hillshade + suitability overlay + landing site + F2 outline."""
    import json
    import rasterio

    outdir = io_utils.resolve_path(cfg, cfg["outputs"]["out"])
    interim = io_utils.resolve_path(cfg, cfg["outputs"]["interim"])
    with rasterio.open(interim / "dem_aoi.tif") as ds:
        dem = ds.read(1); crs = ds.crs; transform = ds.transform
    suit = rasterio.open(outdir / "suitability.tif").read(1)
    hazard = rasterio.open(outdir / "hazard.tif").read(1)

    hs = viz.hillshade(dem, pixel_size_m=cfg["crs"]["pixel_size_m"])
    ax.imshow(hs, cmap="gray", origin="upper")
    suit_go = np.ma.masked_where((hazard > 0) | ~np.isfinite(suit), suit)
    ax.imshow(suit_go, cmap="YlGn", origin="upper", alpha=0.55, vmin=0, vmax=1)

    crater = json.load(open(outdir / "target_crater.geojson"))["features"][0]
    ring = crater["geometry"]["coordinates"][0]
    fc, fr = _ring_to_px(ring, crs, transform)
    ax.plot(fc, fr, color="yellow", lw=1.8)
    ax.annotate("F2", (np.mean(fc), np.mean(fr)), color="yellow", fontsize=11, weight="bold",
                xytext=(6, 6), textcoords="offset points")

    cands = json.load(open(outdir / "landing_candidates.geojson"))["features"]
    r1 = min(cands, key=lambda f: f["properties"]["rank"])
    lon, lat = r1["geometry"]["coordinates"]
    sc, sr = _lonlat_to_px(lon, lat, crs, transform)
    ax.scatter([sc], [sr], marker="*", s=420, c="red", edgecolors="white",
               linewidths=1.3, zorder=5)
    ax.annotate("landing site", (sc, sr), color="white", fontsize=9, weight="bold",
                xytext=(10, -14), textcoords="offset points")
    ax.set_title("Safe landing site (LOLA terrain + illumination)", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])


def _panel_traverse(ax, cfg):
    """(3) Rover route over the detected ice nodes (depot-centred frame)."""
    proc = io_utils.resolve_path(cfg, cfg["dfsar"]["export"]["processed_dir"])
    xs, ys, conf = [], [], []
    with open(proc / "candidates_router.csv") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["x_m"])); ys.append(float(row["y_m"]))
            conf.append(float(row["confidence"]))
    xs, ys, conf = np.array(xs), np.array(ys), np.array(conf)

    # Nearest-neighbour traverse from the depot (origin of the router frame) and back.
    depot = np.array([0.0, 0.0])
    pts = np.column_stack([xs, ys])
    order, cur, remaining = [], depot, list(range(len(pts)))
    while remaining:
        d = [np.hypot(*(pts[i] - cur)) for i in remaining]
        nxt = remaining.pop(int(np.argmin(d)))
        order.append(nxt); cur = pts[nxt]
    route = np.vstack([depot, pts[order], depot])
    ax.plot(route[:, 0], route[:, 1], "-", color="tab:blue", lw=1.6, alpha=0.8, zorder=2)

    # operational range ring (matches the router's ~8 km battery radius)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(8000 * np.cos(th), 8000 * np.sin(th), ls="--", color="0.6", lw=1.0, zorder=1)

    sc = ax.scatter(xs, ys, c=conf, cmap="viridis", s=120, edgecolors="white",
                    linewidths=0.8, zorder=3, vmin=0, vmax=1)
    ax.scatter([0], [0], marker="*", s=460, c="red", edgecolors="white",
               linewidths=1.3, zorder=4)
    ax.annotate("depot", (0, 0), color="black", fontsize=9, weight="bold",
                xytext=(8, 8), textcoords="offset points")
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("ice confidence (CPR/DOP)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    lim = 8800
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([-8000, 0, 8000]); ax.set_yticks([-8000, 0, 8000])
    ax.tick_params(labelsize=7)
    ax.text(0.5, -0.06, "router frame (m), depot at origin", transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="0.3")
    ax.set_title("Rover traverse: 9 ice nodes, 88.9% coverage", fontsize=10.5)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    cfg = io_utils.load_config()
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, left=0.025, right=0.965, top=0.86, bottom=0.16, wspace=0.21)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    _panel_detect(ax1, cfg)
    _panel_land(ax2, cfg)
    _panel_traverse(ax3, cfg)

    fig.canvas.draw()  # finalise positions before reading them for the arrows
    for axl, axr in ((ax1, ax2), (ax2, ax3)):
        pl, pr = axl.get_position(), axr.get_position()
        yv = (pl.y0 + pl.y1) / 2.0
        fig.add_artist(FancyArrowPatch((pl.x1 + 0.006, yv), (pr.x0 - 0.010, yv),
                                       mutation_scale=22, arrowstyle="-|>",
                                       color="0.25", lw=2.4, shrinkA=0, shrinkB=0,
                                       clip_on=False, zorder=10))

    fig.suptitle("Faustini-F2 landing-site pipeline on real Chandrayaan-2 + LOLA data",
                 fontsize=13, weight="bold", y=0.97)
    fig.text(0.5, 0.045,
             "Closed loop on real Faustini-F2 data:  detect → land → traverse.",
             ha="center", fontsize=12, weight="bold")

    out = io_utils.resolve_path(cfg, cfg["outputs"]["out"]) / "pipeline_hero.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Hero figure -> {out}  ({fig.get_size_inches()[0]:.0f}x{fig.get_size_inches()[1]:.0f} in @ 200 dpi)")


if __name__ == "__main__":
    main()
