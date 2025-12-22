import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline


def _get_image_dims(ds):
    """
    Infer image dimensions from longitude/latitude variables,
    excluding 'pol' if present.
    """
    if "longitude" in ds:
        lon_da = ds["longitude"]
    elif "owiLon" in ds:
        lon_da = ds["owiLon"]
    else:
        raise ValueError("originalDataset must contain longitude or owiLon")

    return tuple(d for d in lon_da.dims if d != "pol")


def map_raster(
    raster_ds,
    originalDataset,
    footprint,
    cross_antimeridian=False,
):
    """
    Map a raster onto an image grid defined by originalDataset.

    Parameters
    ----------
    raster_ds : xarray.Dataset or xarray.DataArray
        Raster with valid `.rio` accessor.
    originalDataset : xarray.Dataset
        Dataset defining the target image grid (lon/lat in image dims).
    footprint : shapely.geometry.Polygon
        Footprint of the target grid.
    cross_antimeridian : bool, default False

    Returns
    -------
    xarray.Dataset or xarray.DataArray
    """

    # --- target lon/lat ---
    if "longitude" in originalDataset:
        target_lon = originalDataset["longitude"]
        target_lat = originalDataset["latitude"]
    else:
        target_lon = originalDataset["owiLon"]
        target_lat = originalDataset["owiLat"]

    # --- ensure geographic CRS ---
    if not raster_ds.rio.crs.is_geographic:
        raster_ds = raster_ds.rio.reproject(4326)

    # --- ensure dims ordering ---
    raster_ds = raster_ds.transpose("y", "x")

    # --- lon/lat bounds from footprint ---
    if cross_antimeridian:
        x_vals = np.asarray(footprint.exterior.xy[0]) % 360
        lon_range = [x_vals.min(), x_vals.max()]
        y_vals = np.asarray(footprint.exterior.xy[1])
        lat_range = [y_vals.min(), y_vals.max()]
    else:
        lon1, lat1, lon2, lat2 = footprint.exterior.bounds
        lon_range = [lon1, lon2]
        lat_range = [lat1, lat2]

    # --- ensure increasing raster coords ---
    for coord in ("x", "y"):
        if raster_ds[coord].values[-1] < raster_ds[coord].values[0]:
            raster_ds = raster_ds.reindex({coord: raster_ds[coord][::-1]})

    # --- restrict raster to footprint bbox ---
    ilon_range = [
        max(1, np.searchsorted(raster_ds.x.values, lon_range[0])),
        min(np.searchsorted(raster_ds.x.values, lon_range[1]),
            raster_ds.x.size),
    ]
    ilat_range = [
        max(1, np.searchsorted(raster_ds.y.values, lat_range[0])),
        min(np.searchsorted(raster_ds.y.values, lat_range[1]),
            raster_ds.y.size),
    ]

    ilon_range, ilat_range = [
        [rg[0] - 1, rg[1] + 1] for rg in (ilon_range, ilat_range)
    ]

    raster_ds = raster_ds.isel(
        x=slice(*ilon_range),
        y=slice(*ilat_range),
    )

    # --- intermediate grid size from image dims ---
    az_dim, ra_dim = _get_image_dims(originalDataset)
    ny = originalDataset.sizes[az_dim]
    nx = originalDataset.sizes[ra_dim]

    num = min((ny + nx) // 2, 1000)

    lons = np.linspace(*lon_range, num=num)
    lats = np.linspace(*lat_range, num=num)

    # --- DataArray → Dataset ---
    name = None
    if isinstance(raster_ds, xr.DataArray):
        name = raster_ds.name or "_tmp_name"
        raster_ds = raster_ds.to_dataset(name=name)

    if cross_antimeridian:
        target_lon = target_lon % 360

    mapped = []

    for var in raster_ds:
        da = raster_ds[var]

        # first interpolation step
        if np.any(np.isnan(da.values)):
            upscaled = da.interp(x=lons, y=lats)
        else:
            spline = RectBivariateSpline(
                da.y.values,
                da.x.values,
                da.values,
                kx=3,
                ky=3,
            )
            upscaled = xr.DataArray(
                spline(lats, lons),
                dims=("y", "x"),
                coords={"x": lons, "y": lats},
                name=var,
            )

        # final interpolation on image grid
        mapped.append(
            upscaled
            .interp(x=target_lon, y=target_lat)
            .drop_vars(("x", "y"))
        )

    mapped_ds = xr.merge(mapped)

    # --- Dataset → DataArray ---
    if name is not None:
        mapped_ds = mapped_ds[name]
        if name == "_tmp_name":
            mapped_ds.name = None

    return mapped_ds
