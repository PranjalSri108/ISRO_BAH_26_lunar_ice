"""Runner: 07_export_candidates — export DFSAR ice + landing depot as the router's input.

Turns the dense DFSAR ice mask into a handful of router *candidate nodes* (ice clusters) and
the rank-1 landing site into the rover *depot* (start / recharge point), then writes the CSVs
the path-planner's preprocess.py consumes:

    data/processed/candidates.csv : x, y, conf, weight   (one row per ice cluster node)
    data/processed/depot.csv      : node_type, x, y       (the rank-1 landing site)

All coordinates are projected metres in the DFSAR mosaic CRS (Moon_2000 South Pole
Stereographic) so nodes and depot share one frame. This is a pure hand-off — no routing here
(path planning is a teammate's module, per CLAUDE.md).
"""
from __future__ import annotations

import csv
import json

import numpy as np

import lunar_ice  # noqa: F401
from lunar_ice import io_utils


def _px_to_xy(transform, rows, cols):
    """Pixel (row,col) centres -> projected (x,y) metres."""
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    return x, y


def _geographic_to_mosaic(ds_crs):
    """Transformer from the landing pipeline's lon/lat (Moon sphere) into the mosaic CRS."""
    from pyproj import CRS, Transformer
    return Transformer.from_crs(io_utils.MOON_GEOGRAPHIC, CRS.from_wkt(ds_crs.to_wkt()),
                                always_xy=True)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.transform import rowcol

    cfg = io_utils.load_config()
    d = cfg["dfsar"]
    exp = d["export"]
    outdir = io_utils.resolve_path(cfg, d["out"])
    proc = io_utils.resolve_path(cfg, exp["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)

    # --- load ice rasters -------------------------------------------------------------------
    with rasterio.open(outdir / "ice_mask.tif") as ds:
        ice_mask = ds.read(1)
        transform = ds.transform
        mosaic_crs = ds.crs
    with rasterio.open(outdir / "ice_prob.tif") as ds:
        ice_prob = ds.read(1)

    rows, cols = np.where(ice_mask > 0)
    n_ice = rows.size
    if n_ice == 0:
        raise SystemExit("No ice pixels in ice_mask.tif — run 06_dfsar_detect.py first.")
    xs, ys = _px_to_xy(transform, rows, cols)
    conf_px = ice_prob[rows, cols].astype(float)

    # --- (1) cluster ice pixels into ~n_nodes router nodes ----------------------------------
    from sklearn.cluster import KMeans

    k = int(min(exp["n_nodes"], n_ice))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(
        np.column_stack([xs, ys]))

    nodes = []
    for lab in range(k):
        m = labels == lab
        nodes.append({
            "x": float(xs[m].mean()),
            "y": float(ys[m].mean()),
            "conf": float(conf_px[m].mean()),
            "weight": int(m.sum()),
        })
    nodes.sort(key=lambda n: n["conf"], reverse=True)

    # --- (2) depot = rank-1 landing site, transformed into the mosaic CRS -------------------
    cand = json.load(open(outdir.parent / "landing_candidates.geojson"))
    rank1 = min(cand["features"], key=lambda f: f["properties"]["rank"])
    assert rank1["properties"]["rank"] == 1, "no rank-1 feature in landing_candidates.geojson"
    lon, lat = rank1["geometry"]["coordinates"]
    tf = _geographic_to_mosaic(mosaic_crs)
    depot_x, depot_y = (float(v) for v in tf.transform(lon, lat))

    # --- (3) write CSVs ---------------------------------------------------------------------
    cand_csv = proc / "candidates.csv"
    with open(cand_csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["x", "y", "conf", "weight"])
        wtr.writeheader()
        wtr.writerows(nodes)

    depot_csv = proc / "depot.csv"
    with open(depot_csv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["node_type", "x", "y"])
        wtr.writerow(["depot", depot_x, depot_y])

    # --- (3b) overview PNG ------------------------------------------------------------------
    dists = [float(np.hypot(n["x"] - depot_x, n["y"] - depot_y)) for n in nodes]
    _overview_png(cfg, outdir, mosaic_crs, transform, nodes, (depot_x, depot_y),
                  rowcol, plt, proc / "router_input.png")

    # --- (4) console summary ----------------------------------------------------------------
    confs = [n["conf"] for n in nodes]
    max_d = max(dists)
    print("================= ROUTER EXPORT — SUMMARY =================")
    print(f"Ice pixels clustered      : {n_ice:,} -> {len(nodes)} candidate nodes")
    print(f"Node confidence range     : {min(confs):.3f} .. {max(confs):.3f} "
          f"(mean {np.mean(confs):.3f})")
    print(f"Node weight range (px)    : {min(n['weight'] for n in nodes)} .. "
          f"{max(n['weight'] for n in nodes)}")
    print(f"Depot (rank-1 '{rank1['properties']['profile']}'): "
          f"X={depot_x:.1f} Y={depot_y:.1f} m  (lon={lon:.4f} lat={lat:.4f})")
    print(f"Max node->depot distance  : {max_d/1000:.2f} km  "
          f"(min {min(dists)/1000:.2f} km) -> size router battery/crater_radius accordingly")
    print(f"\nWrote:\n  {cand_csv}\n  {depot_csv}\n  {proc/'router_input.png'}")


def _overview_png(cfg, outdir, mosaic_crs, transform, nodes, depot_xy, rowcol, plt, path):
    """CPR backdrop + ice clusters (sized by conf) + depot + F2 outline, in AOI pixel space."""
    import json as _json
    import rasterio
    from pyproj import CRS, Transformer

    with rasterio.open(outdir / "cpr_aoi.tif") as ds:
        cpr = ds.read(1)
    bg = np.where(np.isfinite(cpr) & (cpr > 0), cpr, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.imshow(bg, cmap="gray", origin="upper", vmin=0,
              vmax=float(np.nanpercentile(bg, 99)))

    # ice cluster nodes -> pixels; size by confidence, colour by confidence
    nrows = [rowcol(transform, n["x"], n["y"])[0] for n in nodes]
    ncols = [rowcol(transform, n["x"], n["y"])[1] for n in nodes]
    confs = np.array([n["conf"] for n in nodes])
    sizes = 40 + 360 * (confs - confs.min()) / (np.ptp(confs) + 1e-9)
    sc = ax.scatter(ncols, nrows, s=sizes, c=confs, cmap="viridis",
                    edgecolors="white", linewidths=0.6, zorder=3, label="ice nodes")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="node confidence (mean ice_prob)")

    # depot
    dr, dc = rowcol(transform, *depot_xy)
    ax.scatter([dc], [dr], marker="*", s=480, c="red", edgecolors="white",
               linewidths=1.2, zorder=4, label="depot (rank-1 site)")

    # F2 outline from target_crater.geojson (lon/lat -> mosaic CRS -> pixels)
    try:
        tc = _json.load(open(outdir.parent / "target_crater.geojson"))
        ring = tc["features"][0]["geometry"]["coordinates"][0]
        tf = Transformer.from_crs(io_utils.MOON_GEOGRAPHIC, CRS.from_wkt(mosaic_crs.to_wkt()),
                                  always_xy=True)
        fxy = [tf.transform(lon, lat) for lon, lat in ring]
        frc = [rowcol(transform, x, y) for x, y in fxy]
        ax.plot([c for _, c in frc], [r for r, _ in frc], color="yellow", lw=1.6,
                zorder=3, label="F2 rim")
    except Exception as e:  # F2 outline is cosmetic; never block the export on it
        print(f"  (F2 outline skipped: {e})")

    ax.set_title("Router input — DFSAR ice clusters + landing depot\n"
                 "(CPR backdrop, F2 AOI; node size/colour = confidence)")
    ax.set_xlabel("col (px)")
    ax.set_ylabel("row (px)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
