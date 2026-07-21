"""Figures and the downstream hand-off bundle.

Hero map (hillshade + balanced suitability + PSR + profile sites + F2), illumination map,
site->F2 cross-sections, trade-off table, and the outputs/ bundle per docs/interface.md
(rasters + geojsons + manifest.json on one shared grid/CRS). Artifacts only — no routes.
"""
from __future__ import annotations

import numpy as np


def hillshade(dem: np.ndarray, pixel_size_m: float = 5.0, azimuth_deg: float = 315.0,
              altitude_deg: float = 45.0) -> np.ndarray:
    """Standard Horn hillshade [0,255] for context backgrounds. NaNs pass through."""
    az = np.deg2rad(360.0 - azimuth_deg + 90.0)
    alt = np.deg2rad(altitude_deg)
    dy, dx = np.gradient(dem, pixel_size_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    shaded = (np.sin(alt) * np.sin(slope)
              + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(shaded, 0, 1) * 255.0


def hero_figure(dem: np.ndarray, suitability: np.ndarray, psr: np.ndarray,
                sites: list[dict], f2: dict, pixel_size_m: float, out_png: str) -> None:
    """Hero map: hillshade + balanced suitability + PSR shade + sites + F2 + lines to F2."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = hillshade(dem, pixel_size_m)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(hs, cmap="gray", origin="upper")
    suit = np.ma.masked_where(suitability <= 0, suitability)
    im = ax.imshow(suit, cmap="viridis", vmin=0, vmax=1, alpha=0.55)
    ax.imshow(np.ma.masked_where(psr == 0, psr), cmap="cool", alpha=0.35)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="balanced suitability [0,1]")

    fr, fc = f2["row"], f2["col"]
    theta = np.linspace(0, 2 * np.pi, 200)
    rr = f2["rim_radius_m"] / pixel_size_m
    ax.plot(fc + rr * np.cos(theta), fr + rr * np.sin(theta), "r-", lw=1.8)
    ax.annotate("F2", (fc, fr), color="red", fontsize=13, fontweight="bold",
                xytext=(6, 6), textcoords="offset points")

    colors = {"safest": "lime", "closest_ice": "cyan", "best_lit": "gold",
              "balanced": "white"}
    for s in sites:
        c = colors.get(s["profile"], "white")
        ax.plot([s["col"], fc], [s["row"], fr], "-", color=c, lw=1.0, alpha=0.7)
        ax.scatter([s["col"]], [s["row"]], s=90, facecolors="none", edgecolors=c,
                   linewidths=2.2)
        ax.annotate(f"{s['profile']} (#{s['rank']})", (s["col"], s["row"]), color=c,
                    fontsize=10, xytext=(8, -12), textcoords="offset points")
    ax.set_title("Landing-site selection near F2 (Faustini) — balanced suitability, "
                 "PSR shaded (cyan)")
    ax.set_xlabel("col (px, 5 m)")
    ax.set_ylabel("row (px, 5 m)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def write_handoff(layer_specs: list[dict], vector_specs: list[dict], out_dir,
                  pixel_size_m: float) -> dict:
    """Copy hand-off rasters to out_dir, validate one shared grid/CRS, write manifest.json.

    layer_specs: dicts with keys name, src, description, units.
    vector_specs: dicts with keys name, description (files already in out_dir).
    Returns the manifest dict.
    """
    import json
    from pathlib import Path

    import rasterio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = None
    manifest = {"crs": None, "pixel_size_m": pixel_size_m, "rasters": [], "vectors": []}

    for spec in layer_specs:
        with rasterio.open(spec["src"]) as ds:
            arr = ds.read(1)
            prof = ds.profile.copy()
            key = (ds.width, ds.height, tuple(ds.transform)[:6], ds.crs.to_string())
        if ref is None:
            ref = key
            manifest["crs"] = key[3]
        elif key != ref:
            raise AssertionError(f"grid/CRS mismatch for {spec['name']}: {key} != {ref}")
        dst = out_dir / spec["name"]
        with rasterio.open(dst, "w", **prof) as ds:
            ds.write(arr, 1)
        nd = prof.get("nodata")
        if nd is None:
            nd_json = None
        elif isinstance(nd, float) and np.isnan(nd):
            nd_json = "nan"  # keep valid JSON; nan is the genuine nodata sentinel
        else:
            nd_json = float(nd)
        manifest["rasters"].append({
            "path": spec["name"], "description": spec["description"],
            "units": spec["units"], "crs": key[3], "pixel_size_m": pixel_size_m,
            "nodata": nd_json,
            "dtype": prof["dtype"], "width": key[0], "height": key[1],
        })
    for spec in vector_specs:
        manifest["vectors"].append({
            "path": spec["name"], "description": spec["description"], "crs": "EPSG:4326"})

    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, allow_nan=False)  # enforce spec-valid JSON
    return manifest
