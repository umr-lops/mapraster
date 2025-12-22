from mapraster.main import _get_image_dims
import numpy as np
import xarray as xr


def test_get_image_dims_ignore_pol():
    ny, nx, npol = 10, 20, 2
    lon = np.zeros((ny, nx, npol))

    ds = xr.Dataset(
        {
            "longitude": (("line", "sample", "pol"), lon),
        }
    )
    az_dim, ra_dim = _get_image_dims(ds)

    assert az_dim == "line"
    assert ra_dim == "sample"


test_get_image_dims_ignore_pol()
