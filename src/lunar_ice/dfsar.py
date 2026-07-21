"""DFSAR subsurface-ice detection over the F2 AOI (PS-8 add-on).

Detects subsurface water ice from the Chandrayaan-2 DFSAR L3C *derived* south-pole-east
mosaic using a Circular-Polarisation-Ratio + volume-scattering criterion:

    ice  <=>  CPR > 1  AND  volume scattering is dominant.

This is the m-chi / Yamaguchi analog of the classic CPR + DOP test: a high CPR caused by
*volume* (not double-bounce / surface) scattering is the radar signature of buried, low-loss
water ice. We assume ~5 m radar penetration depth (L-band).

Everything here is intentionally separate from the LOLA terrain/landing module — per CLAUDE.md
ice detection is OUT of the landing pipeline; this only produces an ice mask + volume estimate
for hand-off. No reprojection: all DFSAR layers share one 25 m polar-stereo grid; the ICY
validation catalogue sits on a sub-pixel-offset grid and is nearest-neighbour aligned here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import io_utils


# --------------------------------------------------------------------------- paths / AOI

def layer_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Resolve every DFSAR raster path from config to an absolute path."""
    d = cfg["dfsar"]
    base = io_utils.resolve_path(cfg, d["dir"])
    paths = {k: base / d[k] for k in ("cpr", "vol", "odd", "evn", "hlx")}
    paths["icy"] = io_utils.resolve_path(cfg, d["icy_mask"])
    return paths


def aoi_bounds(cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    """+/- half_width_m box around F2 in DFSAR projected metres (left, bottom, right, top)."""
    fx, fy = cfg["dfsar"]["f2_xy"]
    hw = float(cfg["dfsar"]["half_width_m"])
    return (fx - hw, fy - hw, fx + hw, fy + hw)


# --------------------------------------------------------------------------- reading

def read_aoi(cfg: dict[str, Any]) -> dict[str, Any]:
    """Window-read the AOI from every DFSAR layer (never the full mosaic).

    Returns a dict with each band array (cpr/vol/odd/evn/hlx), the catalogue ``icy`` aligned
    onto the CPR AOI grid, the shared ``profile`` (CPR window), and the F2 (row, col) in AOI.
    """
    from rasterio.transform import rowcol

    bounds = aoi_bounds(cfg)
    out: dict[str, Any] = {}
    profile = None
    for key in ("cpr", "vol", "odd", "evn", "hlx"):
        arr, prof = io_utils.window_read(layer_paths(cfg)[key], bounds)
        out[key] = arr.astype(np.float64)
        if profile is None:
            profile = prof
    out["profile"] = profile

    # ICY catalogue lives on a sub-pixel-offset grid -> align (nearest) to the CPR AOI grid.
    out["icy"] = align_to_profile(layer_paths(cfg)["icy"], bounds, profile)

    fx, fy = cfg["dfsar"]["f2_xy"]
    row, col = rowcol(profile["transform"], fx, fy)
    out["f2_rowcol"] = (int(row), int(col))
    return out


def align_to_profile(src_path, bounds, ref_profile) -> np.ndarray:
    """Read ``src_path`` over ``bounds`` and resample (nearest) onto ``ref_profile``'s grid.

    Same CRS / resolution as the reference, but a small sub-pixel grid offset — so a plain
    window read would be off by up to a pixel. We force src/dst CRS equal (the catalogue CRS
    string is 'unknown' though physically identical) and warp by nearest neighbour.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    src_arr, src_prof = io_utils.window_read(src_path, bounds)
    dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_prof["transform"],
        src_crs=ref_profile["crs"],
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=Resampling.nearest,
    )
    return dst


# --------------------------------------------------------------------------- detection

@dataclass
class IceResult:
    """Everything the detection produces over the AOI."""

    valid: np.ndarray            # bool: finite + non-zero mosaic power
    cpr: np.ndarray
    vol_total: np.ndarray        # odd + evn + hlx + vol
    vol_frac: np.ndarray         # vol / vol_total (0..1; >0.5 => volume dominant)
    vol_dominant: np.ndarray     # bool: vol > odd + evn + hlx  (primary definition)
    cpr_high: np.ndarray         # bool: cpr > cpr_thresh
    ice_mask: np.ndarray         # uint8: cpr_high & vol_dominant & valid  (primary)
    ice_mask_alt: np.ndarray     # uint8: cpr_high & (vol > AOI p90) & valid (alternate)
    ice_prob: np.ndarray         # float32 [0,1]: how strongly BOTH conditions hold
    vol_p90: float
    meta: dict = field(default_factory=dict)


def detect_ice(bands: dict[str, Any], cfg: dict[str, Any]) -> IceResult:
    """Apply the CPR + volume-dominant ice criterion to the AOI bands."""
    d = cfg["dfsar"]
    cpr = bands["cpr"]
    vol, odd, evn, hlx = bands["vol"], bands["odd"], bands["evn"], bands["hlx"]

    surface = odd + evn + hlx                # non-volume (surface + double-bounce) power
    total = surface + vol

    # Mask mosaic nodata / zeros: a pixel is usable only if all powers are finite and the
    # total scattered power is > 0 (zero-power pixels are mosaic gaps, not real returns).
    finite = (np.isfinite(cpr) & np.isfinite(vol) & np.isfinite(odd)
              & np.isfinite(evn) & np.isfinite(hlx))
    valid = finite & (total > 0) & (cpr > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        vol_frac = np.where(total > 0, vol / total, 0.0)

    cpr_thresh = float(d["cpr_thresh"])
    cpr_ref = float(d["cpr_ref"])
    cpr_high = valid & (cpr > cpr_thresh)
    vol_dominant = valid & (vol > surface)            # vol_frac > 0.5

    ice_mask = (cpr_high & vol_dominant).astype(np.uint8)

    # Alternate vol-dominant definition: VOL above the AOI's own high-percentile.
    vol_valid = vol[valid]
    vol_p90 = float(np.percentile(vol_valid, float(d["vol_pctile"]))) if vol_valid.size else 0.0
    ice_mask_alt = (cpr_high & valid & (vol > vol_p90)).astype(np.uint8)

    # ice_prob: how strongly BOTH conditions hold (product -> needs both).
    #   cpr strength : (cpr - thresh) / (cpr_ref - thresh), clipped to [0,1]
    #   vol strength : (vol_frac - 0.5) / 0.5, clipped to [0,1]  (0 at the 0.5 boundary)
    cpr_strength = np.clip((cpr - cpr_thresh) / max(cpr_ref - cpr_thresh, 1e-9), 0.0, 1.0)
    vol_strength = np.clip((vol_frac - 0.5) / 0.5, 0.0, 1.0)
    ice_prob = np.where(valid, cpr_strength * vol_strength, 0.0).astype(np.float32)

    return IceResult(
        valid=valid, cpr=cpr, vol_total=total, vol_frac=vol_frac,
        vol_dominant=vol_dominant, cpr_high=cpr_high,
        ice_mask=ice_mask, ice_mask_alt=ice_mask_alt, ice_prob=ice_prob,
        vol_p90=vol_p90,
        meta={"cpr_thresh": cpr_thresh, "cpr_ref": cpr_ref,
              "vol_pctile": float(d["vol_pctile"])},
    )


def f2_interior_mask(profile, f2_rowcol, radius_m: float, pixel_size_m: float) -> np.ndarray:
    """Boolean disk of radius ``radius_m`` around F2 — a PROXY for the crater interior."""
    h, w = profile["height"], profile["width"]
    r0, c0 = f2_rowcol
    rr, cc = np.ogrid[:h, :w]
    dist = np.hypot((rr - r0), (cc - c0)) * pixel_size_m
    return dist <= radius_m


# --------------------------------------------------------------------------- validation

def validate(ice_mask: np.ndarray, icy_catalogue: np.ndarray,
             valid: np.ndarray) -> dict[str, float]:
    """Compare the detected mask against the ICY_CRATERS catalogue over the AOI.

    Returns agreement (% of detected pixels also in catalogue) and recovery (% of catalogue
    ice we recover), restricted to valid mosaic pixels.
    """
    mine = (ice_mask > 0) & valid
    cat = (icy_catalogue > 0.5) & valid
    tp = int(np.sum(mine & cat))
    n_mine = int(np.sum(mine))
    n_cat = int(np.sum(cat))
    return {
        "catalogue_px_in_aoi": n_cat,
        "detected_px": n_mine,
        "true_positive_px": tp,
        "agreement_pct": 100.0 * tp / n_mine if n_mine else 0.0,   # of mine, how many are catalogued
        "recovery_pct": 100.0 * tp / n_cat if n_cat else None,     # of catalogue, how much recovered
    }


# --------------------------------------------------------------------------- volume / mass

def _eps_birchak(phi: float, eps_solid: float, eps_ice: float) -> float:
    """Birchak / CRIM (refractive) mixing, pore space fully ice-filled: sqrt(eps) is volume-weighted."""
    return ((1.0 - phi) * np.sqrt(eps_solid) + phi * np.sqrt(eps_ice)) ** 2


def _eps_maxwell_garnett(phi: float, eps_solid: float, eps_ice: float) -> float:
    """Maxwell-Garnett: ice spheres (fraction phi) embedded in a regolith matrix (eps_solid)."""
    em, ei, f = eps_solid, eps_ice, phi
    num = ei + 2 * em + 2 * f * (ei - em)
    den = ei + 2 * em - f * (ei - em)
    return em * num / den


def estimate_volume(ice_px: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Ice volume + water-equivalent mass for low/central/high porosity.

    ice volume = (mask area) x (penetration depth) x (ice fraction), with the pore-filled
    assumption ice_fraction = porosity. Maxwell-Garnett & Birchak effective permittivities are
    reported alongside as the dielectric cross-check (volume scattering <-> low-loss ice).
    """
    d = cfg["dfsar"]
    px = float(d["pixel_size_m"])
    depth = float(d["penetration_depth_m"])
    rho = float(d["ice_density_kgm3"])
    eps_s, eps_i = float(d["eps_solid"]), float(d["eps_ice"])

    area_m2 = ice_px * px * px
    rows = {}
    for level, phi in d["porosity"].items():        # low / central / high
        phi = float(phi)
        ice_frac = phi                               # pore space fully ice-filled (stated assumption)
        vol_m3 = area_m2 * depth * ice_frac
        mass_kg = vol_m3 * rho
        rows[level] = {
            "porosity": phi,
            "ice_fraction": ice_frac,
            "ice_area_km2": area_m2 / 1e6,
            "depth_m": depth,
            "ice_volume_m3": vol_m3,
            "water_mass_kg": mass_kg,
            "water_mass_Mt": mass_kg / 1e9,          # 1 Mt = 1e9 kg
            "eps_birchak": _eps_birchak(phi, eps_s, eps_i),
            "eps_maxwell_garnett": _eps_maxwell_garnett(phi, eps_s, eps_i),
        }
    return {"area_m2": area_m2, "area_km2": area_m2 / 1e6, "levels": rows}


# ===========================================================================================
# Full-polarimetric CPR + DOP (true Stokes) — for 08_fp_detect.py
# ===========================================================================================
# The derived-mosaic path above uses CPR + decomposition (a proxy for volume scattering). The
# functions below run the *real* full-pol test on a complex SLI scene: form the multilooked
# Stokes vector from the [HH HV; VH VV] scattering matrix (standard linear-basis formulation)
# and apply CPR > 1 AND DOP < 0.13 (Sinha et al. 2026, author-confirmed). Low DOP = depolarised
# = volume scattering = the subsurface-ice signature; high DOP would be ordinary surface return.


def linear_stokes_terms(hh, hv, vh, vv):
    """Per-pixel intensity / covariance terms that, once multilooked, give Stokes S1..S4.

    HH/HV/VH/VV are complex SLI samples. HV/VH are averaged (reciprocity). The Stokes vector is
    built in the standard linear (H/V) basis:

        S1 = <|HH|^2> + 2<|HV|^2> + <|VV|^2>      (total power / span)
        S2 = <|HH|^2> - <|VV|^2>
        S3 =  2 Re<HH . conj(VV)>
        S4 = -2 Im<HH . conj(VV)>

    where <.> is the 3x3 boxcar ensemble average. We return the FIVE real terms to multilook
    BEFORE forming Stokes (averaging the single-look terms is what makes DOP < 1 meaningful):

        ( |HH|^2 , |HV|^2 , |VV|^2 , Re(HH.conj(VV)) , Im(HH.conj(VV)) )
    """
    shv = 0.5 * (hv + vh)                 # HV/VH reciprocity
    c = hh * np.conj(vv)
    return ((hh * np.conj(hh)).real, (shv * np.conj(shv)).real,
            (vv * np.conj(vv)).real, c.real, c.imag)


def stokes_from_terms(i_hh, i_hv, i_vv, re_hhvv, im_hhvv):
    """Assemble Stokes S1..S4 from the (already multilooked) linear-basis terms."""
    s1 = i_hh + 2.0 * i_hv + i_vv
    s2 = i_hh - i_vv
    s3 = 2.0 * re_hhvv
    s4 = -2.0 * im_hhvv
    return s1, s2, s3, s4


def boxcar_multilook(arr, look: int):
    """3x3 (look x look) boxcar average then subsample by ``look`` (the multilook operation).

    Edges use 'nearest' so window seams don't bias. Subsampling takes the centre of each
    look-sized group (offset look//2) so the phase is consistent across azimuth windows.
    """
    from scipy.ndimage import uniform_filter

    sm = uniform_filter(arr.astype(np.float64), size=look, mode="nearest")
    off = look // 2
    return sm[off::look, off::look]


def stokes_to_cpr_dop(s1, s2, s3, s4):
    """CPR = (S1 - S4)/(S1 + S4) and DOP = sqrt(S2^2+S3^2+S4^2)/S1, guarding S1<=0."""
    good = s1 > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        cpr = np.where(good, (s1 - s4) / (s1 + s4), np.nan)
        dop = np.where(good, np.sqrt(s2 * s2 + s3 * s3 + s4 * s4) / s1, np.nan)
    return cpr, dop


def fp_ice(cpr, dop, valid, cfg):
    """Full-pol ice mask + probability: CPR > cpr_thresh AND DOP < dop_thresh.

    ice_prob = how strongly BOTH hold: clip((CPR-thr)/(cpr_ref-thr)) * clip((dop_thr-DOP)/dop_thr).
    """
    f = cfg["fp"]
    cpr_thr, cpr_ref = float(f["cpr_thresh"]), float(f["cpr_ref"])
    dop_thr = float(f["dop_thresh"])
    finite = valid & np.isfinite(cpr) & np.isfinite(dop)
    ice_mask = (finite & (cpr > cpr_thr) & (dop < dop_thr)).astype(np.uint8)
    cpr_strength = np.clip((cpr - cpr_thr) / max(cpr_ref - cpr_thr, 1e-9), 0.0, 1.0)
    dop_strength = np.clip((dop_thr - dop) / max(dop_thr, 1e-9), 0.0, 1.0)
    ice_prob = np.where(finite, cpr_strength * dop_strength, 0.0).astype(np.float32)
    return ice_mask, ice_prob


def read_gsli_gcps(csv_path, sli_height: int, sli_width: int):
    """Parse the g_sli geometry CSV into (row, col, lon, lat) ground-control points.

    The CSV is a regular azimuth-major grid: ``n_az`` lines x ``n_rg`` range samples, columns
    Latitude, Longitude, Slant_Range, Incidence. ``n_rg`` is auto-detected from the slant-range
    reset period (range increases across a line, then resets at the next azimuth line). Grid
    node (i, j) maps linearly onto the SLI as row = i*(H-1)/(n_az-1), col = j*(W-1)/(n_rg-1).
    Returns (rows, cols, lons, lats) as float arrays in original SLI pixel coordinates.
    """
    lat, lon, sr = [], [], []
    with open(csv_path) as fh:
        next(fh)                                   # header
        for line in fh:
            p = line.split(",")
            if len(p) < 3:
                continue
            lat.append(float(p[0])); lon.append(float(p[1])); sr.append(float(p[2]))
    lat = np.asarray(lat); lon = np.asarray(lon); sr = np.asarray(sr)

    # range count = index of the first slant-range reset (sr decreases) + 1
    drops = np.where(np.diff(sr) < 0)[0]
    n_rg = int(drops[0] + 1) if drops.size else sr.size
    n_az = lat.size // n_rg
    if n_az * n_rg != lat.size:
        lat, lon = lat[: n_az * n_rg], lon[: n_az * n_rg]   # trim any partial trailing line

    i = np.repeat(np.arange(n_az), n_rg)           # azimuth index per sample
    j = np.tile(np.arange(n_rg), n_az)             # range index per sample
    rows = i * (sli_height - 1) / max(n_az - 1, 1)
    cols = j * (sli_width - 1) / max(n_rg - 1, 1)
    return rows, cols, lon, lat, (n_az, n_rg)
