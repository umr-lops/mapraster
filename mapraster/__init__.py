# type: ignore[attr-defined]
"""mapraster is a Python lib to interpolate xarray raster field on image geometry (e.g. line/sample)"""


__all__ = ["map_raster"]

from .main import map_raster

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata
try:
    __version__ = metadata.version("mapraster")
except Exception:
    __version__ = "999"
