import xarray as xr
import numpy as np
import yaml
import geopandas as gpd
import pandas as pd


def coarsen_and_reduce(da: xr.DataArray, veg_type: int, **kwargs) -> xr.DataArray:
    """execute `.coarsen` and `.reduce`, with args handled
    by this function, due to xarray bug (described below).

    Combined, these functions allow the raw veg raster to be
    downsampled into an individual array containing pct cover
    of a single constituent vegetation classes.

    :param (xr.DataArray) da: DataArray containing single-band veg raster
    :param (int) veg_type: single vegetation type to use as a pct cover
        value for the new, downsampled veg array.

    :return (xr.DataArray): Downsampled array with pct cover for a SINGLE veg type.
    """

    def _count_vegtype_and_calculate_percentage(block, axis):
        """Get percentage cover of vegetation types in pixel group

        NOTE: This function is nested as a work-around to the kwargs bug
        in DataArray.reduce, described here:
        https://github.com/pydata/xarray/issues/8059. This avoids having
        to use a global variable to provide args to the `.reduce` call.


        :param (np.ndarray) block: non-overlapping chunks from `.coarsen` function.
        :param (tuple) axis: used to index the chunks, which are returned with dims
            based on the coarsen dims.
        :return (np.ndarray): coarsened chunk
        """
        # Sum over the provided axis
        count = (block == veg_type).sum(axis=axis)
        total_cells = block.shape[axis[0]] * block.shape[axis[1]]
        return (count / total_cells) * 100

    result = da.coarsen(**kwargs).reduce(_count_vegtype_and_calculate_percentage)

    return result


def generate_pct_cover(
    data_array: xr.DataArray, veg_keys: pd.DataFrame, **kwargs
) -> None:
    """iterate vegetation types, merge all arrays, serialize to NetCDF.

    The creates a pct cover .nc for each veg types list in the veg_keys
    dataframe.

    :param (xr.DataArray) data_array: input array
    :param (pd.DataFrame) veg_keys: df with veg types. Must
        include col with title "Values". This could be generalized
        to a list in future.

    :return: None, output is .nc file
    """
    veg_types = veg_keys["Value"].values
    veg_arrays = []

    for n, i in enumerate(veg_types):
        print(f"processing veg type: {i}, (number {n} out of {len(veg_types)})")
        new_da = coarsen_and_reduce(data_array, veg_type=i, **kwargs)
        new_da = new_da.rename(f"pct_cover_{i}")
        veg_arrays.append(new_da)

    ds_out = xr.merge(veg_arrays)
    ds_out.to_netcdf("./pct_cover.nc")


def generate_pct_cover_custom(da: xr.DataArray, veg_types: list, **kwargs):
    """Generate pct cover for combinations of veg types

    Uses `coarsen_and_reduce` with an intermediate array with bools
    for desired veg type. TODO: update code to separate CLI methdods,
    i.e. `python module.py` from imported module.

    :param (xr.DataArray) data_array: input array; must include x,y dims
    :param (list) veg_types: List of veg types to consider as "True"
        for percent cover

    :return: None, output is .nc file
    """
    # create new binary var with tuple of dims, data
    da["binary"] = (["x", "y"], np.isin(da, veg_types))
    # run coarsen w/ True as valid veg type
    da_out = coarsen_and_reduce(da=da["binary"], veg_type=True, **kwargs)
    da_out.to_netcdf("./pct_cover.nc")


def read_veg_key(path: str) -> pd.DataFrame:
    """load vegetation class names from database file"""
    dbf = gpd.read_file(path)
    # fix dtype
    dbf["Value"] = dbf["Value"].astype(int)
    return dbf


if __name__ == "__main__":
    CONFIG_PATH = "./VegProcessor/veg_config.yaml"
    # Load the YAML configuration file
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    # Assign variables from the YAML file
    veg_raster_path = config.pop("veg_raster_path")
    x_dim_coarsen = config.pop("x_dim")
    y_dim_coarsen = config.pop("y_dim")
    veg_keys_path = config.pop("veg_keys_path")

    # load veg raster
    ds = xr.open_dataset(
        veg_raster_path,
        engine="rasterio",
    )
    # subset to dataarray
    da = ds["band_data"]["band" == 0]

    # load veg keys
    veg_keys_df = read_veg_key(veg_keys_path)
    # create .nc with layers for each veg class, dims must be passed to .coarsen
    generate_pct_cover(da, veg_keys=veg_keys_df, x=10, y=10, boundary="trim")