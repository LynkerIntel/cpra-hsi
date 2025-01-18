from osgeo import gdal

import xarray as xr
from geopandas import GeoDataFrame
import rioxarray

import collections
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio.features
import scipy.interpolate
from scipy import ndimage as nd
from skimage.measure import label
from skimage.measure import find_contours
from shapely.geometry import LineString, MultiLineString, shape

# from datacube.helpers import
from datacube.utils.geometry import CRS, Geometry

# convenience function from
# https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Frequently_used_code/Rasterise_vectorise.html

# adds xarray-oriented wrapper to rasterio functionality,
# useful for zonal stats, where zones are defined by a geopandas df


def xr_rasterize(
    gdf,
    da,
    attribute_col=False,
    crs=None,
    transform=None,
    name=None,
    x_dim="x",
    y_dim="y",
    export_tiff=None,
    **rasterio_kwargs,
):
    """
    Rasterizes a geopandas.GeoDataFrame into an xarray.DataArray.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A geopandas.GeoDataFrame object containing the vector/shapefile
        data you want to rasterise.
    da : xarray.DataArray
        The shape, coordinates, dimensions, and transform of this object
        are used to build the rasterized shapefile. It effectively
        provides a template. The attributes of this object are also
        appended to the output xarray.DataArray.
    attribute_col : string, optional
        Name of the attribute column in the geodataframe that the pixels
        in the raster will contain.  If set to False, output will be a
        boolean array of 1's and 0's.
    crs : str, optional
        CRS metadata to add to the output xarray. e.g. 'epsg:3577'.
        The function will attempt get this info from the input
        GeoDataFrame first.
    transform : affine.Affine object, optional
        An affine.Affine object (e.g. `from affine import Affine;
        Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "6886890.0) giving the
        affine transformation used to convert raster coordinates
        (e.g. [0, 0]) to geographic coordinates. If none is provided,
        the function will attempt to obtain an affine transformation
        from the xarray object (e.g. either at `da.transform` or
        `da.geobox.transform`).
    x_dim : str, optional
        An optional string allowing you to override the xarray dimension
        used for x coordinates. Defaults to 'x'.
    y_dim : str, optional
        An optional string allowing you to override the xarray dimension
        used for y coordinates. Defaults to 'y'.
    export_tiff: str, optional
        If a filepath is provided (e.g 'output/output.tif'), will export a
        geotiff file. A named array is required for this operation, if one
        is not supplied by the user a default name, 'data', is used
    **rasterio_kwargs :
        A set of keyword arguments to rasterio.features.rasterize
        Can include: 'all_touched', 'merge_alg', 'dtype'.

    Returns
    -------
    xarr : xarray.DataArray

    """

    # Check for a crs object
    try:
        crs = da.crs
    except:
        if crs is None:
            raise Exception(
                "Please add a `crs` attribute to the "
                "xarray.DataArray, or provide a CRS using the "
                "function's `crs` parameter (e.g. 'EPSG:3577')"
            )

    # Check if transform is provided as a xarray.DataArray method.
    # If not, require supplied Affine
    if transform is None:
        try:
            # First, try to take transform info from geobox
            transform = da.geobox.transform
        # If no geobox
        except:
            try:
                # Try getting transform from 'transform' attribute
                transform = da.transform
            except:
                # If neither of those options work, raise an exception telling the
                # user to provide a transform
                raise Exception(
                    "Please provide an Affine transform object using the "
                    "`transform` parameter (e.g. `from affine import "
                    "Affine; Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "
                    "6886890.0)`"
                )

    # Get the dims, coords, and output shape from da
    da = da.squeeze()
    y, x = da.shape
    dims = list(da.dims)
    xy_coords = [da[y_dim], da[x_dim]]

    # Reproject shapefile to match CRS of raster
    print(
        f"Rasterizing to match xarray.DataArray dimensions ({y}, {x}) "
        f"and projection system/CRS (e.g. {crs})"
    )

    try:
        gdf_reproj = gdf.to_crs(crs=crs)
    except:
        # sometimes the crs can be a datacube utils CRS object
        # so convert to string before reprojecting
        gdf_reproj = gdf.to_crs(crs={"init": str(crs)})

    # If an attribute column is specified, rasterise using vector
    # attribute values. Otherwise, rasterise into a boolean array
    if attribute_col:
        # Use the geometry and attributes from `gdf` to create an iterable
        shapes = zip(gdf_reproj.geometry, gdf_reproj[attribute_col].astype(int))

        # Convert polygons into a numpy array using attribute values
        arr = rasterio.features.rasterize(
            shapes=shapes, out_shape=(y, x), transform=transform, **rasterio_kwargs
        )
    else:
        # Convert polygons into a boolean numpy array
        arr = rasterio.features.rasterize(
            shapes=gdf_reproj.geometry,
            out_shape=(y, x),
            transform=transform,
            **rasterio_kwargs,
        )

    # Convert result to a xarray.DataArray
    if name is not None:
        xarr = xr.DataArray(arr, coords=xy_coords, dims=dims, attrs=da.attrs, name=name)
    else:
        xarr = xr.DataArray(arr, coords=xy_coords, dims=dims, attrs=da.attrs)

    # add back crs if da.attrs doesn't have it
    if "crs" not in xarr.attrs:
        xarr.attrs["crs"] = str(crs)

    if export_tiff:
        try:
            print("Exporting GeoTIFF with array name: " + name)
            ds = xarr.to_dataset(name=name)
            # xarray bug removes metadata, add it back
            ds[name].attrs = xarr.attrs
            ds.attrs = xarr.attrs
            write_geotiff(export_tiff, ds)

        except:
            print("Exporting GeoTIFF with default array name: 'data'")
            ds = xarr.to_dataset(name="data")
            ds.data.attrs = xarr.attrs
            ds.attrs = xarr.attrs
            write_geotiff(export_tiff, ds)

    return xarr


def create_wpu_raster(
    veg_raster_path: str,
    gdf: GeoDataFrame,
    output_raster_path: str,
    attribute_col: str = "WPU_ID",
    crs: str = "EPSG:32615",
):
    """
    Create a WPU (Water Planning Unit) zones raster from a vegetation raster and GeoDataFrame.

    Parameters
    ----------
    veg_raster_path : str
        Path to the input vegetation raster file.
    gdf : GeoDataFrame
        A GeoDataFrame containing WPU zone boundaries and attributes.
    output_raster_path : str
        Path to save the output WPU raster.
    attribute_col : str, optional
        Column in the GeoDataFrame that contains the WPU ID. Default is 'WPU_ID'.
    crs : str, optional
        Coordinate Reference System (CRS) for the output raster. Default is 'EPSG:32615'.

    Returns
    -------
    None
        Saves the rasterized WPU zones to the specified output path.
    """
    # Load the input vegetation raster
    veg_da = xr.open_dataarray(veg_raster_path)
    veg_da = veg_da.isel(band=0)  # Select the first band if multiple bands exist
    veg_da.rio.write_crs(crs, inplace=True)

    # Rasterize the GeoDataFrame to align with the vegetation raster
    resampled_wpu = xr_rasterize(
        gdf=gdf, da=veg_da, attribute_col=attribute_col, crs=crs
    )

    # Save the result as a GeoTIFF with appropriate compression
    resampled_wpu.astype("int32").rio.to_raster(
        output_raster_path, driver="GTiff", compress="LZW"
    )

    print(f"Saved WPU raster to {output_raster_path}")
