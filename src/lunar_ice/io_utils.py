"""I/O, config, and CRS helpers — the shared plumbing for the pipeline.

Everything else in :mod:`lunar_ice` reads paths/params through :func:`load_config`
and locates F2 / windows through the coordinate helpers here. Rasters are always
read/written with their CRS + nodata preserved (see CLAUDE.md conventions).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pyproj import CRS, Transformer

# Repo root = two levels up from this file (src/lunar_ice/io_utils.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"

# F2 / inputs live in a south-polar-stereographic frame on a 1737.4 km sphere.
# lon/lat are reported in plain geographic coords on that same sphere (no datum shift).
MOON_GEOGRAPHIC = CRS.from_proj4(
    "+proj=longlat +a=1737400 +b=1737400 +no_defs"
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config/config.yaml (the single source of truth) into a dict."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(REPO_ROOT)
    return cfg


def resolve_path(cfg: dict[str, Any], rel: str | Path) -> Path:
    """Resolve a config-relative path against the repo root."""
    return (REPO_ROOT / Path(rel)).resolve()


def dem_crs(cfg: dict[str, Any]) -> CRS:
    """The working CRS (south polar stereographic) from config."""
    return CRS.from_proj4(cfg["crs"]["proj4"])


def lonlat_to_xy(lon: float, lat: float, cfg: dict[str, Any]) -> tuple[float, float]:
    """Convert geographic lon/lat (deg) on the Moon sphere into DEM CRS metres."""
    tf = Transformer.from_crs(MOON_GEOGRAPHIC, dem_crs(cfg), always_xy=True)
    x, y = tf.transform(lon, lat)
    return float(x), float(y)


def xy_to_lonlat(x: float, y: float, cfg: dict[str, Any]) -> tuple[float, float]:
    """Inverse of :func:`lonlat_to_xy` — DEM metres back to lon/lat (deg)."""
    tf = Transformer.from_crs(dem_crs(cfg), MOON_GEOGRAPHIC, always_xy=True)
    lon, lat = tf.transform(x, y)
    return float(lon), float(lat)


@dataclass
class RasterMeta:
    """Lightweight raster header (metadata only — no pixel data loaded)."""

    path: str
    crs: str
    bounds: tuple[float, float, float, float]  # left, bottom, right, top
    width: int
    height: int
    pixel_size: tuple[float, float]  # (x_res, y_res) in CRS units
    nodata: float | None
    dtype: str


def raster_meta(path: str | Path) -> RasterMeta:
    """Open a raster read-only and return its header. Reads no pixels."""
    import rasterio

    with rasterio.open(path) as ds:
        b = ds.bounds
        return RasterMeta(
            path=str(path),
            crs=ds.crs.to_string() if ds.crs else "<none>",
            bounds=(b.left, b.bottom, b.right, b.top),
            width=ds.width,
            height=ds.height,
            pixel_size=(abs(ds.transform.a), abs(ds.transform.e)),
            nodata=ds.nodata,
            dtype=ds.dtypes[0],
        )


def point_in_bounds(x: float, y: float, bounds: tuple[float, float, float, float]) -> bool:
    """True iff (x, y) lies within (left, bottom, right, top)."""
    left, bottom, right, top = bounds
    return left <= x <= right and bottom <= y <= top


def window_read(path: str | Path, bounds: tuple[float, float, float, float]):
    """Window-read ONLY the pixels covering `bounds` (left, bottom, right, top).

    Never loads the full raster. Returns (array, profile) where profile carries the
    windowed transform, the source CRS, nodata, and dtype — ready for :func:`write_raster`.
    """
    import rasterio
    from rasterio.windows import from_bounds, transform as window_transform

    left, bottom, right, top = bounds
    with rasterio.open(path) as ds:
        win = from_bounds(left, bottom, right, top, ds.transform)
        # Snap to whole pixels so the AOI grid stays aligned to the source grid.
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        arr = ds.read(1, window=win)
        profile = ds.profile.copy()
        profile.update(
            height=arr.shape[0],
            width=arr.shape[1],
            transform=window_transform(win, ds.transform),
        )
    return arr, profile


def write_raster(path: str | Path, arr, profile: dict) -> None:
    """Write a single-band raster, preserving CRS / transform / nodata / dtype."""
    import rasterio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = profile.copy()
    out.update(count=1, dtype=str(arr.dtype))
    with rasterio.open(path, "w", **out) as ds:
        ds.write(arr, 1)
