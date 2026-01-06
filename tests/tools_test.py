import numpy as np
import rioxarray  # activate .rio accessor
import xarray as xr
from shapely.geometry import Polygon


def fake_dataset(cross_antimeridian=False):
    nline, nsample = 50, 60

    line = np.arange(nline)
    sample = np.arange(nsample)

    L, S = np.meshgrid(line, sample, indexing="ij")

    if cross_antimeridian:
        # near +180°, wrapped to [-180, 180]
        lon_raw = 170 + 0.25 * S - 0.12 * L
        lon = ((lon_raw + 180) % 360) - 180
    else:
        # same geometry, far from ±180°
        lon = -30 + 0.25 * S - 0.12 * L

    # slightly tilted latitude grid
    lat = -29 + 0.06 * L + 0.01 * S

    return xr.Dataset(
        data_vars={
            "longitude": (("line", "sample"), lon),
            "latitude": (("line", "sample"), lat),
        },
        coords={"line": line, "sample": sample},
    )


def _to_lon180(ds):
    # convert [0, 360] → [-180, 180]
    ds = ds.roll(x=-np.searchsorted(ds.x, 180), roll_coords=True)
    ds["x"] = xr.where(ds["x"] >= 180, ds["x"] - 360, ds["x"])
    return ds


def _to_lon360(ds):
    # ensure [0, 360] longitude
    ds = ds.assign_coords(x=ds.x % 360)
    return ds.sortby("x")


def fake_ecmwf_0100_1h(*, to180=True, with_nan=False):
    import datetime

    lon = np.linspace(0, 360, 360 * 10, endpoint=False)
    lat = np.linspace(-90, 90, 181 * 10)

    LON, LAT = np.meshgrid(lon, lat)

    U10 = 5 + 2 * np.cos(np.deg2rad(LAT))
    V10 = 2 * np.sin(np.deg2rad(LON))

    ds = xr.Dataset(
        data_vars={
            "U10": (("y", "x"), U10),
            "V10": (("y", "x"), V10),
        },
        coords={"x": lon, "y": lat},
        attrs={"time": datetime.datetime(2023, 3, 3, 7)},
    )

    ds = _to_lon180(ds) if to180 else _to_lon360(ds)

    if with_nan:
        # latitude band
        lat_mask = (ds["y"] >= -40) & (ds["y"] <= -20)

        if to180:
            lon_mask_1 = (ds["x"] >= 170) & (ds["x"] <= 179)
            lon_mask_2 = (ds["x"] >= -179) & (ds["x"] <= -170)
        else:
            lon_mask_1 = (ds["x"] >= 170) & (ds["x"] <= 179)
            lon_mask_2 = (ds["x"] >= 180) & (ds["x"] <= 189)

        zone = (lat_mask & lon_mask_1) | (lat_mask & lon_mask_2)

        iy = xr.DataArray(
            np.arange(ds.sizes["y"]),
            dims=("y",),
            coords={"y": ds["y"]},
        )
        ix = xr.DataArray(
            np.arange(ds.sizes["x"]),
            dims=("x",),
            coords={"x": ds["x"]},
        )

        sparse_mask = ((iy % 2) == 0) & ((ix % 3) == 0)
        final_mask = zone & sparse_mask

        ds["U10"] = ds["U10"].where(~final_mask)
        ds["V10"] = ds["V10"].where(~final_mask)

    ds.rio.write_crs("EPSG:4326", inplace=True)
    return ds


def build_footprint(ds):
    lon = ds["longitude"].values
    lat = ds["latitude"].values

    return Polygon(
        [
            (lon[0, 0], lat[0, 0]),
            (lon[0, -1], lat[0, -1]),
            (lon[-1, -1], lat[-1, -1]),
            (lon[-1, 0], lat[-1, 0]),
        ]
    )
