"""Multi-criteria suitability + hard hazard mask.

Normalize each criterion to [0, 1] (slope/roughness/relief lower=better,
ice_proximity nearer-F2=better but excluding the crater interior, illumination
higher=better), build a uint8 hazard no-go mask from the hard constraints, then a
weighted suitability raster per profile (=0 inside hazard). All transparent, no ML.
"""
from __future__ import annotations

import numpy as np


def distance_to_point(shape: tuple[int, int], transform, x0: float, y0: float) -> np.ndarray:
    """Euclidean distance (m) from every pixel CENTRE to point (x0, y0) in CRS units."""
    h, w = shape
    xs = transform.c + (np.arange(w) + 0.5) * transform.a
    ys = transform.f + (np.arange(h) + 0.5) * transform.e
    dx = xs[None, :] - x0
    dy = ys[:, None] - y0
    return np.hypot(dx, dy).astype(np.float32)


def estimate_crater_radius(dem: np.ndarray, transform, x0: float, y0: float,
                           search_m: float = 3000.0, n_az: int = 72,
                           step_m: float = 20.0) -> float:
    """Estimate F2's rim radius (m): azimuthal-median radius of the elevation crest.

    Marches radially out from the crater centre in n_az directions, takes the radius of
    peak elevation (the rim) per azimuth, and returns the median across azimuths.
    """
    h, w = dem.shape
    radii = np.arange(step_m, search_m + step_m, step_m)
    rim_radii = []
    for theta in np.linspace(0.0, 2 * np.pi, n_az, endpoint=False):
        ux, uy = np.cos(theta), np.sin(theta)
        cols = ((x0 + radii * ux - transform.c) / transform.a).astype(int)
        rows = ((y0 + radii * uy - transform.f) / transform.e).astype(int)
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        if not ok.any():
            continue
        prof = dem[rows[ok], cols[ok]]
        rim_radii.append(radii[ok][int(np.argmax(prof))])
    return float(np.median(rim_radii)) if rim_radii else 0.0


def normalize_low_good(arr: np.ndarray, hi: float) -> np.ndarray:
    """Lower=better: clip(1 - arr/hi, 0, 1). Value hits 0 at the hazard threshold `hi`."""
    return np.clip(1.0 - arr.astype(np.float32) / hi, 0.0, 1.0)


def ice_proximity(dist_to_f2_m: np.ndarray, inner_m: float, max_m: float) -> np.ndarray:
    """Nearer F2 = better; 0 inside `inner_m` (crater+rim buffer) and beyond `max_m`."""
    prox = (max_m - dist_to_f2_m) / (max_m - inner_m)
    prox = np.clip(prox, 0.0, 1.0)
    prox[dist_to_f2_m < inner_m] = 0.0
    return prox.astype(np.float32)


def auto_threshold(arr: np.ndarray, configured, pct: float = 90.0) -> float:
    """Use the configured value, or the `pct`-th percentile of finite data if null."""
    if configured is not None:
        return float(configured)
    finite = arr[np.isfinite(arr)]
    return float(np.percentile(finite, pct))


def build_hazard(slope, roughness, relief, dist_to_f2_m, inner_excl_m: float,
                 cfg: dict) -> tuple[np.ndarray, dict]:
    """uint8 {0,1} no-go: steep OR rough OR high-relief OR inside F2 rim+buffer OR too far.

    Returns (hazard, thresholds) where thresholds documents the values used.
    """
    c = cfg["constraints"]
    slope_max = float(c["slope_max_deg"])
    rough_max = auto_threshold(roughness, c.get("roughness_max"))
    relief_max = auto_threshold(relief, c.get("relief_max_m"))
    max_dist = float(cfg["landing"]["max_dist_from_F2_m"])

    hazard = (
        (slope > slope_max)
        | (roughness > rough_max)
        | (relief > relief_max)
        | (dist_to_f2_m < inner_excl_m)
        | (dist_to_f2_m > max_dist)
        | ~np.isfinite(slope)
    ).astype(np.uint8)
    thresholds = {
        "slope_max_deg": slope_max,
        "roughness_max": rough_max,
        "relief_max_m": relief_max,
        "inner_excl_m": inner_excl_m,
        "max_dist_from_F2_m": max_dist,
    }
    return hazard, thresholds


def weighted_suitability(criteria: dict[str, np.ndarray], weights: dict[str, float],
                         hazard: np.ndarray) -> np.ndarray:
    """Weighted mean of normalized criteria in [0,1], forced to 0 inside hazard."""
    wsum = sum(weights[k] for k in weights)
    score = np.zeros_like(next(iter(criteria.values())), dtype=np.float32)
    for k, wk in weights.items():
        score += wk * criteria[k]
    score /= wsum
    score[hazard == 1] = 0.0
    return np.clip(score, 0.0, 1.0).astype(np.float32)
