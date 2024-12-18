import xarray as xr
import pathlib
import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime


def generate_combined_sequence(
    quintile_sequence: pd.DataFrame,
    quintile_to_year_map: dict[int, int],
    source_folder: str,  # Path where HEC-RAS .tif files are stored
    output_folder: str,  # Path to store the combined 25-year sequence
):
    """
    Generate a 25-year sequence of HEC-RAS .tif files based on quintile assignments. Currently
    requires raster input to be MONTHLY timeseries.

    Parameters
    ----------
    quintile_sequence : pd.DataFrame
        The 25-year sequence of quintiles.
    quintile_to_year_map : dict[int, int])
        Maps quintiles to available years (e.g., {1: 2006, 2: 2023}).
    source_folder : str
        Path where HEC-RAS .tif files are stored.
    output_folder : str
        Path to store the combined 25-year sequence.

    Returns
    --------
    Saves WSE data with new filenames to simulate 25 year model output from analog years.
    """
    os.makedirs(output_folder, exist_ok=True)
    path = pathlib.Path(source_folder)

    source_files = [
        file for file in list(path.rglob("*.tif")) if "SLR" not in str(file)
    ]

    if not source_files:
        raise FileNotFoundError(f"No .tif files found for year in {source_folder}")

    # Loop through the 25-year quintile sequence
    for _, i in quintile_sequence.iterrows():
        analog_year = int(i["Water Year"])
        source_year = quintile_to_year_map[i.Quintile]
        start = datetime(source_year - 1, 10, 1)
        end = datetime(source_year, 9, 30)

        source_year_paths = [
            path for path in source_files if start <= extract_date(path) <= end
        ]
        source_year_paths.sort()

        print(f"Mapping {source_year} to {analog_year}")

        if len(source_year_paths) < 12:
            raise ValueError(f"Missing data in source files for {source_year}.")

        for p in source_year_paths:
            print(f"input path: {p}")
            # for each monthly file, copy and rename to analog year
            file_date = extract_date(p)
            # October 1 is the start of the water year
            if file_date.month in [10, 11, 12]:
                replacement_year = str(int(analog_year) - 1)
            else:  # Before October, it's the previous water year
                replacement_year = analog_year

            # Reconstruct the string with the new water year (with zero padding)
            new_file_name = f"WSE_MEAN_{replacement_year}_{file_date.month:02d}_{file_date.day:02d}.tif"
            print(f"output name: {new_file_name}")

            dest_file = os.path.join(output_folder, new_file_name)
            shutil.copy(p, dest_file)

        print("All months completed.")

    print(f"Generated 25-year sequence in {output_folder}")
    print("WARN: only files NAMES were modified, original timestamps still in place.")


def extract_date(path: pathlib.Path) -> datetime:
    """
    Extract date from HEC-RAS filepaths, or any filepath with dates in a YYYY_MM_DD format.
    Must use pathlib object not str.

    Parameters
    ----------
    path : pathlib.Path
        Path to the file with a date embedded in its name in the format YYYY_MM_DD.

    Returns
    -------
    datetime or None
        The extracted date as a datetime object, or None if no valid date is found.
    """
    try:
        # Assuming the date is located just before the file extension in YYYY_MM_DD format
        date_str = path.stem.split("_")[-3:]  # Extract the last three
        date_str = "_".join(date_str)  # Combine back into "YYYY_MM_DD"
        return datetime.strptime(date_str, "%Y_%m_%d")
    except (ValueError, IndexError):
        return None


def create_dataset_from_template(
    template: xr.Dataset, new_variables: dict[np.ndarray, str]
) -> xr.Dataset:
    """
    Create an xarray.Dataset based on a template dataset.

    Parameters
    ----------
    template : xr.Dataset
        The template dataset containing dimensions, coordinates, and optional global attributes.
    new_variables : dict
        Dictionary defining new variables. Keys are variable names, and values are tuples (data, attrs).
        - `data`: NumPy array of the same shape as the template's data variables.
        - `attrs`: Metadata for the variable.

    Returns
    -------
    xr.Dataset
        A new dataset based on the template, containing the new variables and copied template attributes.
    """
    coords = {name: template.coords[name] for name in template.coords}
    new_ds = xr.Dataset(coords=coords)

    for var_name, (data, attrs) in new_variables.items():
        # Check that the shape matches the template
        if data.shape != template["WSE_MEAN"].shape:
            raise ValueError(
                f"Shape of variable '{var_name}' ({data.shape}) does not match "
                f"the template shape ({template['WSE_MEAN'].shape})."
            )
        # Add the variable
        new_ds[var_name] = xr.DataArray(
            data, dims=template["WSE_MEAN"].dims, attrs=attrs
        )

    # Optionally, copy global attributes from the template
    new_ds.attrs = template.attrs

    return new_ds


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


def load_mf_tifs(
    wse_directory_path: str,
    variable_name: str = "WSE_MEAN",
    date_format: str = "%Y_%m_%d",
) -> xr.Dataset:
    """Load a folder of .tif files, each representing a timestep, into an xarray.Dataset.

    Each .tif file should have a filename that includes a variable name (e.g., `WSE_MEAN`)
    followed by a timestamp in the specified format (e.g., `2005_10_01`). The function will
    automatically extract the timestamps and assign them to a 'time' dimension in the resulting
    xarray.Dataset.

    Naming of the method may change if other model outputs require a totally different function
    or can be adapter with flexible args to this method.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the .tif files.
    variable_name : str, optional
        The name of the variable to use in the dataset.
    date_format : str, optional
        Format string for parsing dates from file names, default is "%Y_%m_%d".
        Adjust based on your file naming convention.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset with the raster data from each .tif file stacked along
        a 'time' dimension, with the specified variable name.
    """
    tif_files = sorted(glob.glob(os.path.join(wse_directory_path, "*.tif")))

    time_stamps = [
        pd.to_datetime(
            "_".join(os.path.basename(f).split("_")[2:5]).replace(".tif", ""),
            format=date_format,
        )
        for f in tif_files
    ]

    # preprocess function to remove the 'band' dimension
    def preprocess(da):
        return da.squeeze(dim="band").expand_dims(
            time=[time_stamps[tif_files.index(da.encoding["source"])]]
        )

    xr_dataset = xr.open_mfdataset(
        tif_files,
        concat_dim="time",
        combine="nested",
        parallel=True,
        preprocess=preprocess,
    )

    # convert data variable keys to a list and rename the main variable
    xr_dataset = xr_dataset.rename(
        {list(xr_dataset.data_vars.keys())[0]: variable_name}
    )

    return xr_dataset


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
