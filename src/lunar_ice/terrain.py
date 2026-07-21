"""Terrain derivatives over the AOI: roughness, local relief, curvature.

Inputs are the AOI crops (dem_aoi, slope_aoi). Roughness here is a BOULDER PROXY
(no optical/OHRC data available) — see CLAUDE.md. Window-read only; never the full DEM.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def win_px(win_m: float, pixel_size_m: float) -> int:
    """Window size in pixels for a metric defined over win_m metres (>= 1)."""
    return max(1, int(round(win_m / pixel_size_m)))


def roughness(slope: np.ndarray, win: int) -> np.ndarray:
    """Local std of slope (deg) in a `win`-pixel window — a BOULDER PROXY.

    True boulder detection needs optical/OHRC imagery, absent here (see CLAUDE.md).
    Computed as sqrt(E[s^2] - E[s]^2) via uniform (box) filters.
    """
    s = slope.astype(np.float64)
    mean = ndimage.uniform_filter(s, size=win, mode="nearest")
    mean_sq = ndimage.uniform_filter(s * s, size=win, mode="nearest")
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return np.sqrt(var).astype(np.float32)


def local_relief(dem: np.ndarray, win: int) -> np.ndarray:
    """Local elevation range (max - min, metres) in a `win`-pixel window."""
    d = dem.astype(np.float32)
    hi = ndimage.maximum_filter(d, size=win, mode="nearest")
    lo = ndimage.minimum_filter(d, size=win, mode="nearest")
    return (hi - lo).astype(np.float32)


def curvature(dem: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Laplacian surface curvature (1/m): convex >0, concave <0. Per metre^2 units."""
    lap = ndimage.laplace(dem.astype(np.float64))
    return (lap / (pixel_size_m * pixel_size_m)).astype(np.float32)
