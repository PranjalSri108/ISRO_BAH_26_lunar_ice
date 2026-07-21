"""Runner: 04_candidates — Safe go-set, ellipse test, ranked sites per profile. See docs/BUILD_PROMPTS.md.

Compute the SAFE go-set once (a disc of ellipse_radius_m entirely within slope/roughness limits),
then per profile pick the top non-overlapping site = max mean-suitability over its ellipse, within
min/max distance of F2. Report quantified metrics per site and write landing_candidates.geojson,
landing_site_polygon.geojson, target_crater.geojson. Prints the comparison table.
"""
from __future__ import annotations

import numpy as np

import lunar_ice  # noqa: F401
from lunar_ice import candidates as cd
from lunar_ice import io_utils, suitability as su


def _read(interim, name):
    import rasterio
    with rasterio.open(interim / name) as ds:
        return ds.read(1), ds.profile.copy()


def _stamp(shape, row, col, radius_px):
    """Boolean disc footprint of given radius centred at (row, col) within `shape`."""
    h, w = shape
    r0, r1 = max(0, row - radius_px), min(h, row + radius_px + 1)
    c0, c1 = max(0, col - radius_px), min(w, col + radius_px + 1)
    yy, xx = np.ogrid[r0 - row:r1 - row, c0 - col:c1 - col]
    out = np.zeros(shape, dtype=bool)
    out[r0:r1, c0:c1] = (yy * yy + xx * xx) <= radius_px * radius_px
    return out


def _rc(transform, x, y):
    from rasterio.transform import rowcol
    return rowcol(transform, x, y)


def main() -> None:
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import Point
    from shapely.ops import transform as shp_transform

    cfg = io_utils.load_config()
    interim = io_utils.resolve_path(cfg, cfg["outputs"]["interim"])
    out_dir = io_utils.resolve_path(cfg, cfg["outputs"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    px = float(cfg["crs"]["pixel_size_m"])

    slope, profile = _read(interim, "slope_aoi.tif")
    dem, _ = _read(interim, "dem_aoi.tif")
    roughness, _ = _read(interim, "roughness.tif")
    relief, _ = _read(interim, "relief.tif")
    illum, _ = _read(interim, "illumination_index.tif")
    hazard, _ = _read(interim, "hazard.tif")
    transform = profile["transform"]
    H, W = dem.shape

    # F2 geometry (same derivation as P4).
    fx, fy = io_utils.lonlat_to_xy(cfg["target"]["lon"], cfg["target"]["lat"], cfg)
    R_f2 = su.estimate_crater_radius(dem, transform, fx, fy)
    inner_excl = R_f2 + float(cfg["landing"]["min_dist_from_F2_rim_m"])
    max_dist = float(cfg["landing"]["max_dist_from_F2_m"])
    dist = su.distance_to_point((H, W), transform, fx, fy)

    slope_max = float(cfg["constraints"]["slope_max_deg"])
    rough_max = su.auto_threshold(roughness, cfg["constraints"].get("roughness_max"))

    # ---- SAFE go-set: disc of ellipse_radius_m entirely within slope/roughness limits ----
    r_ell = int(round(float(cfg["landing"]["ellipse_radius_m"]) / px))
    safety_mask = (slope <= slope_max) & (roughness <= rough_max) & np.isfinite(slope)
    goset = cd.safe_goset(safety_mask, r_ell)
    band = (dist >= inner_excl) & (dist <= max_dist)
    eligible = goset & band
    print(f"ellipse radius = {cfg['landing']['ellipse_radius_m']} m ({r_ell}px); "
          f"F2 rim ~{R_f2:.0f}m, band [{inner_excl:.0f}, {max_dist:.0f}] m")
    print(f"safe go-set (ellipse test): {100*goset.mean():.2f}% of AOI; "
          f"eligible (in distance band): {100*eligible.mean():.2f}%  "
          f"({eligible.sum():,} px)")
    if not eligible.any():
        raise SystemExit("No eligible sites — enlarge AOI or relax constraints.")

    ell_suit = {p: cd.ellipse_mean(_read(interim, f"suitability_{p}.tif")[0], r_ell)
                for p in cfg["site_profiles"]}

    rough_sorted = np.sort(roughness[np.isfinite(roughness)])
    r_1km = int(round(1000.0 / px))

    # ---- Pick one top, non-overlapping site per profile ----
    taken = np.zeros((H, W), dtype=bool)
    sites = []
    for profile_name in cfg["site_profiles"]:
        field = ell_suit[profile_name]
        cand = eligible & ~taken
        scored = np.where(cand, field, -1.0)
        row, col = np.unravel_index(int(np.argmax(scored)), scored.shape)
        taken |= _stamp((H, W), row, col, 2 * r_ell)  # non-overlap exclusion

        sl = cd.disc_values(slope, row, col, r_ell)
        ro = cd.disc_values(roughness, row, col, r_ell)
        re = cd.disc_values(relief, row, col, r_ell)
        il = cd.disc_values(illum, row, col, r_ell)
        go_nb = cd.disc_values((hazard == 0).astype(np.float32), row, col, r_1km)
        rough_mean = float(np.mean(ro))
        rough_pct = 100.0 * np.searchsorted(rough_sorted, rough_mean) / rough_sorted.size
        d_km = float(dist[row, col]) / 1000.0
        illum_mean = float(np.mean(il))
        x = transform.c + (col + 0.5) * transform.a
        y = transform.f + (row + 0.5) * transform.e
        lon, lat = io_utils.xy_to_lonlat(x, y, cfg)

        rationale = (
            f"{profile_name}: mean slope {sl.mean():.1f}deg (max {sl.max():.1f}), "
            f"roughness {rough_pct:.0f}th pct, relief {re.mean():.1f}m, "
            f"{d_km:.1f}km from F2, lit~{100*illum_mean:.0f}%, "
            f"{100*go_nb.mean():.0f}% go within 1km")
        sites.append({
            "profile": profile_name, "row": int(row), "col": int(col),
            "x": float(x), "y": float(y), "lon": float(lon), "lat": float(lat),
            "score": float(field[row, col]),
            "slope_mean": float(sl.mean()), "slope_max": float(sl.max()),
            "roughness_pct": float(rough_pct), "relief_m": float(re.mean()),
            "dist_to_F2_km": d_km, "illum_index": illum_mean,
            "lit_pct": float(100 * illum_mean), "pct_go_1km": float(100 * go_nb.mean()),
            "rationale": rationale,
        })

    order = sorted(range(len(sites)), key=lambda i: -sites[i]["score"])
    for rank, i in enumerate(order, 1):
        sites[i]["rank"] = rank

    print("\n" + "=" * 90)
    print(f"{'profile':12s}{'rank':>5s}{'score':>7s}{'slpMean':>8s}{'slpMax':>7s}"
          f"{'roPct':>6s}{'relief':>7s}{'distKm':>7s}{'lit%':>6s}{'go1km%':>7s}")
    print("-" * 90)
    for s in sites:
        print(f"{s['profile']:12s}{s['rank']:>5d}{s['score']:>7.3f}"
              f"{s['slope_mean']:>8.2f}{s['slope_max']:>7.2f}{s['roughness_pct']:>6.0f}"
              f"{s['relief_m']:>7.1f}{s['dist_to_F2_km']:>7.2f}{s['lit_pct']:>6.0f}"
              f"{s['pct_go_1km']:>7.0f}")
    print("=" * 90)
    for s in sites:
        print("  - " + s["rationale"])

    # ---- GeoJSON outputs (lon/lat; labelled EPSG:4326 per docs/interface.md) ----
    props = ["profile", "rank", "score", "slope_mean", "slope_max", "roughness_pct",
             "relief_m", "dist_to_F2_km", "illum_index", "lit_pct", "pct_go_1km",
             "rationale"]
    gpd.GeoDataFrame(
        [{k: s[k] for k in props} for s in sites],
        geometry=[Point(s["lon"], s["lat"]) for s in sites], crs="EPSG:4326"
    ).to_file(out_dir / "landing_candidates.geojson", driver="GeoJSON")

    to_lonlat = Transformer.from_crs(io_utils.dem_crs(cfg), io_utils.MOON_GEOGRAPHIC,
                                     always_xy=True)

    def circle_lonlat(cx, cy, radius_m):
        return shp_transform(lambda X, Y, z=None: to_lonlat.transform(X, Y),
                             Point(cx, cy).buffer(radius_m, resolution=64))

    best = sites[order[0]]
    ell_poly = circle_lonlat(best["x"], best["y"], float(cfg["landing"]["ellipse_radius_m"]))
    gpd.GeoDataFrame([{"profile": best["profile"], "rank": best["rank"],
                       "radius_m": float(cfg["landing"]["ellipse_radius_m"]),
                       "score": best["score"]}],
                     geometry=[ell_poly], crs="EPSG:4326").to_file(
        out_dir / "landing_site_polygon.geojson", driver="GeoJSON")

    f2_poly = circle_lonlat(fx, fy, R_f2)
    fr, fc = _rc(transform, fx, fy)
    floor_il = float(np.mean(cd.disc_values(illum, fr, fc, int(R_f2 / px))))
    gpd.GeoDataFrame([{"name": cfg["target"]["name"], "lat": cfg["target"]["lat"],
                       "lon": cfg["target"]["lon"], "rim_radius_m": R_f2,
                       "floor_illum_index": floor_il}],
                     geometry=[f2_poly], crs="EPSG:4326").to_file(
        out_dir / "target_crater.geojson", driver="GeoJSON")

    print(f"\nwrote: {out_dir/'landing_candidates.geojson'} ({len(sites)} sites)")
    print(f"wrote: {out_dir/'landing_site_polygon.geojson'} (best={best['profile']})")
    print(f"wrote: {out_dir/'target_crater.geojson'} (F2 rim ~{R_f2:.0f} m)")


if __name__ == "__main__":
    main()
