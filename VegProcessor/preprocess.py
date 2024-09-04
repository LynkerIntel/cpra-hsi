import xarray as xr
import numpy as np
import yaml


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
        count_ones = (block == veg_type).sum(axis=axis)
        total_cells = block.shape[axis[0]] * block.shape[axis[1]]
        return (count_ones / total_cells) * 100

    result = da.coarsen(**kwargs).reduce(_count_vegtype_and_calculate_percentage)

    return result


def generate_pct_cover_arrays(data_array: xr.DataArray, **kwargs) -> xr.Dataset:
    """ """
    veg_types = [1, 2, 3]
    veg_arrays = []

    for i in veg_types:
        new_da = coarsen_and_reduce(data_array, veg_type=i, **kwargs)
        new_da = new_da.rename(f"pct_cover_{i}")
        veg_arrays.append(new_da)

    ds_out = xr.merge(veg_arrays)
    # add to dataset?
    # save individual layers?
    return ds_out


if __name__ == "__main__":
    CONFIG_PATH = "../VegProcessor/veg_config.yaml"
    # Load the YAML configuration file
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    # Assign variables from the YAML file
    veg_raster_path = config.pop("veg_raster_path")
    x_dim_coarsen = config.pop("x_dim")
    y_dim_coarsen = config.pop("y_dim")

    # load veg raster
    ds = xr.open_dataset(
        veg_raster_path,
        engine="rasterio",
    )
    # subset to dataarray
    da = ds["band_data"]["band" == 0]

    out = generate_pct_cover_arrays(da, x=10, y=10, boundary="trim")
