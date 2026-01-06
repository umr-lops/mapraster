import numpy as np
import xarray as xr
from tools_test import build_footprint, fake_dataset, fake_ecmwf_0100_1h

from mapraster.main import _get_image_dims, map_raster


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


def test_data_type():
    """
    Make sure that map_raster works when input is DataArray or Dataset
    """

    dataset = fake_dataset(cross_antimeridian=False)
    footprint = build_footprint(dataset)
    raster = fake_ecmwf_0100_1h(
        to180=True,
        with_nan=False,
    )

    assert (
        type(
            map_raster(
                raster_ds=raster,  # Dataset
                originalDataset=dataset,
                footprint=footprint,
                cross_antimeridian=False,
            )
        )
        == xr.Dataset
    )

    assert (
        type(
            map_raster(
                raster_ds=raster.U10,  # DataArray
                originalDataset=dataset,
                footprint=footprint,
                cross_antimeridian=False,
            )
        )
        == xr.DataArray
    )


test_data_type()
