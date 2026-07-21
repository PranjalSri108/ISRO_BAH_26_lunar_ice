"""Candidate landing sites: safe go-set, landing-ellipse test, quantified justification.

A pixel is eligible only if a contiguous disc of ellipse_radius_m around it is ENTIRELY
within slope/roughness limits (the landing-ellipse test). For each profile pick the top
site by mean suitability over its ellipse, within min/max distance of F2 and
non-overlapping, then report numeric justification per site.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def disc(radius_px: int) -> np.ndarray:
    """Boolean disc structuring element of the given pixel radius."""
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (x * x + y * y) <= radius_px * radius_px


def safe_goset(safety_mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Pixels whose FULL disc of radius_px is within the safety mask (ellipse test).

    Implemented via the Euclidean distance transform (exact, O(N)): a safe pixel whose
    nearest unsafe pixel is >= radius_px away has an all-safe disc around it.
    """
    dist_to_unsafe = ndimage.distance_transform_edt(safety_mask)
    return dist_to_unsafe >= radius_px


def ellipse_mean(field: np.ndarray, radius_px: int) -> np.ndarray:
    """Mean of `field` over a disc of radius_px at every pixel (the ellipse footprint)."""
    k = disc(radius_px).astype(np.float32)
    k /= k.sum()
    return ndimage.convolve(field.astype(np.float32), k, mode="nearest")


def disc_values(field: np.ndarray, row: int, col: int, radius_px: int) -> np.ndarray:
    """Values of `field` inside the disc centred at (row, col), clipped to bounds."""
    h, w = field.shape
    r0, r1 = max(0, row - radius_px), min(h, row + radius_px + 1)
    c0, c1 = max(0, col - radius_px), min(w, col + radius_px + 1)
    sub = field[r0:r1, c0:c1]
    yy, xx = np.ogrid[r0 - row:r1 - row, c0 - col:c1 - col]
    m = (yy * yy + xx * xx) <= radius_px * radius_px
    return sub[m]
