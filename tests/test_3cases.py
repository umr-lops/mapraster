import numpy as np
import xarray as xr
from tools_test import build_footprint, fake_dataset, fake_ecmwf_0100_1h

from mapraster.main import map_raster


def test_map_raster_no_antimeridian():
    dataset = fake_dataset(cross_antimeridian=False)
    footprint = build_footprint(dataset)

    raster = fake_ecmwf_0100_1h(
        to180=True,
        with_nan=False,
    )

    out = map_raster(
        raster_ds=raster,
        originalDataset=dataset,
        footprint=footprint,
        cross_antimeridian=False,
    )

    assert set(out.data_vars) == {"U10", "V10"}
    assert out["U10"].shape == dataset["longitude"].shape
    assert not np.all(np.isnan(out["U10"].values))


test_map_raster_no_antimeridian()


def test_map_raster_cross_antimeridian():
    dataset = fake_dataset(cross_antimeridian=True)
    footprint = build_footprint(dataset)

    raster = fake_ecmwf_0100_1h(
        to180=False,
        with_nan=False,
    )

    out = map_raster(
        raster_ds=raster,
        originalDataset=dataset,
        footprint=footprint,
        cross_antimeridian=True,
    )

    assert set(out.data_vars) == {"U10", "V10"}
    assert out["U10"].shape == dataset["longitude"].shape
    assert not np.all(np.isnan(out["V10"].values))


test_map_raster_cross_antimeridian()


def test_map_raster_with_nan():
    dataset = fake_dataset(cross_antimeridian=True)
    footprint = build_footprint(dataset)

    raster = fake_ecmwf_0100_1h(
        to180=False,
        with_nan=True,
    )

    out = map_raster(
        raster_ds=raster,
        originalDataset=dataset,
        footprint=footprint,
        cross_antimeridian=True,
    )

    # No full NaN since raster is build with partial NaN only
    assert not np.all(np.isnan(out["U10"].values))

    # check there are NaN in the output
    nan_ratio = np.isnan(out["U10"].values).mean()
    assert nan_ratio > 0.01


test_map_raster_with_nan()
