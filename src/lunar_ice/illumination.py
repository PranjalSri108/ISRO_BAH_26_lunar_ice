"""Horizon-based illumination index + PSR mask (the credibility upgrade).

Method (a PROXY, not a modelled sun ephemeris):
  1. Block-mean the 5 m DEM down to compute_res_m for tractability.
  2. For each of n_azimuths directions, compute the per-pixel HORIZON ELEVATION ANGLE:
     march outward to max_horizon_m and keep the running max of atan((z(d) - z0) / d).
     Marching is a vectorised shift-and-fmax over the whole grid (no per-pixel loop),
     with a dense-near / sparse-far distance schedule.
  3. illumination_index = mean over azimuths of
     clip((sun_elev_max_deg - horizon_deg) / sun_elev_max_deg, 0, 1)  -> [0, 1].
     At the lunar south pole the Sun never rises above ~1.53 deg, so a pixel whose
     horizon meets/exceeds that in every azimuth is never lit (PSR, index ~ 0); a pixel
     with a low horizon all around is lit a large fraction of the year (index -> 1).

This is an annual-sunlit-FRACTION proxy from topography alone — see CLAUDE.md / docs.
"""
from __future__ import annotations

import numpy as np


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen by an integer factor via block mean (NaN-aware). Crops to a multiple."""
    h = (arr.shape[0] // factor) * factor
    w = (arr.shape[1] // factor) * factor
    a = arr[:h, :w].astype(np.float64)
    a = a.reshape(h // factor, factor, w // factor, factor)
    return np.nanmean(a, axis=(1, 3)).astype(np.float32)


def _step_schedule(max_steps: int, n_far: int = 80) -> np.ndarray:
    """Distance steps (in pixels): every pixel near the origin, geometric far out."""
    near = np.arange(1, min(50, max_steps) + 1)
    if max_steps > 50:
        far = np.geomspace(51, max_steps, n_far)
        steps = np.unique(np.round(np.concatenate([near, far])).astype(int))
    else:
        steps = near
    return steps[steps <= max_steps]


def _shift_nan(z: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r, c] = z[r + dr, c + dc], NaN where the source falls outside the grid."""
    h, w = z.shape
    out = np.full_like(z, np.nan)
    out[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)] = \
        z[max(0, dr):h - max(0, -dr), max(0, dc):w - max(0, -dc)]
    return out


def illumination_index(dem: np.ndarray, res_m: float, n_azimuths: int,
                       max_horizon_m: float, sun_elev_max_deg: float) -> np.ndarray:
    """Horizon-based annual-sunlit-fraction proxy in [0, 1] on the input (coarse) grid."""
    z0 = dem.astype(np.float32)
    max_steps = max(1, int(round(max_horizon_m / res_m)))
    steps = _step_schedule(max_steps)
    azimuths = np.deg2rad(np.linspace(0.0, 360.0, n_azimuths, endpoint=False))

    accum = np.zeros_like(z0, dtype=np.float64)
    for theta in azimuths:
        ux, uy = np.cos(theta), np.sin(theta)
        horizon_tan = np.full_like(z0, -np.inf)
        for k in steps:
            dr, dc = int(round(k * uy)), int(round(k * ux))
            if dr == 0 and dc == 0:
                continue
            dist = np.hypot(dr * res_m, dc * res_m)
            shifted = _shift_nan(z0, dr, dc)
            horizon_tan = np.fmax(horizon_tan, (shifted - z0) / dist)
        horizon_deg = np.degrees(np.arctan(horizon_tan))  # -inf -> -90 (open sky)
        lit = np.clip((sun_elev_max_deg - horizon_deg) / sun_elev_max_deg, 0.0, 1.0)
        accum += lit
    return (accum / n_azimuths).astype(np.float32)


def psr_mask(index: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Permanently-shadowed proxy (uint8): index ~ 0 = never lit in any azimuth."""
    return (index <= eps).astype(np.uint8)
