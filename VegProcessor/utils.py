import xarray as xr
import pathlib
from scipy import ndimage
import scipy.stats as stats

import numpy as np
import pandas as pd
from xrspatial.zonal import crosstab
import yaml

import os
import shutil
from datetime import datetime
from typing import Callable, List, Tuple
import logging
import re
from pathlib import Path

# Configure the logger in VegTransition
logger = logging.getLogger("VegTransition")


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

    source_files = list(path.rglob("*.tif"))

    if not source_files:
        raise FileNotFoundError(
            f"No .tif files found for year in {source_folder}"
        )

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
            raise ValueError(
                f"Missing data in source files for {source_year}."
            )

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
    print(
        "WARN: only files NAMES were modified, original timestamps still in place."
    )


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


def coarsen_and_reduce(
    da: xr.DataArray, veg_type: int | bool, **kwargs
) -> xr.DataArray:
    """execute `.coarsen` and `.reduce`, with args handled
    by this function, due to xarray bug (described below).

    Combined, these functions allow the raw veg raster to be
    downsampled into an individual array containing pct cover
    of a single constituent vegetation classes.

    :param (xr.DataArray) da: DataArray containing single-band veg raster
    :param (int | bool) veg_type: single vegetation type to use as a pct cover
        value for the new, downsampled veg array. Can also be boolean for
        treating multiple veg types as True.

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

    result = da.coarsen(**kwargs).reduce(
        _count_vegtype_and_calculate_percentage
    )

    return result


def generate_pct_cover(
    data_array: xr.DataArray, veg_keys: pd.DataFrame, **kwargs
) -> xr.Dataset:
    """iterate vegetation types, merge all arrays, serialize to NetCDF.

    The creates a pct cover .nc for each veg types list in the veg_keys
    dataframe.

    :param (xr.DataArray) data_array: input array
    :param (pd.DataFrame) veg_keys: df with veg types. Must
        include col with title "Values". This could be generalized
        to a list in future.

    :return (xr.Dataset): ds with layers for each veg type
    """
    veg_types = veg_keys["Value"].values
    veg_arrays = []

    for n, i in enumerate(veg_types):
        logging.info(
            "processing veg type: %s, (number %s out of %s)", i, n, veg_types
        )
        new_da = coarsen_and_reduce(data_array, veg_type=i, **kwargs)
        new_da = new_da.rename(f"pct_cover_{i}")
        veg_arrays.append(new_da)

    ds_out = xr.merge(veg_arrays)
    # ds_out.to_netcdf("./pct_cover.nc")
    return ds_out


def generate_pct_cover_custom(
    data_array: xr.DataArray, veg_types: list, **kwargs
):
    """Generate pct cover for combinations of veg types

    Uses `coarsen_and_reduce` with an intermediate array with bools
    for desired veg type.

    :param (xr.DataArray) data_array: input array; must include x,y dims
    :param (list) veg_types: List of veg types to consider as "True"
        for percent cover

    :return: Aggregated xr.DataArray
    """
    # create new binary var with tuple of dims, data
    data_array["boolean"] = (["y", "x"], np.isin(data_array, veg_types))
    # run coarsen w/ True as valid veg type
    da_out = coarsen_and_reduce(
        da=data_array["boolean"], veg_type=True, **kwargs
    )
    # da_out.to_netcdf("./pct_cover.nc")
    return da_out


def read_veg_key(path: str) -> pd.DataFrame:
    """load vegetation class names from database file"""
    dbf = gpd.read_file(path)
    # fix dtype
    dbf["Value"] = dbf["Value"].astype(int)
    return dbf


def find_nan_to_true_values(
    array1: np.ndarray, array2: np.ndarray, lookup_array: np.ndarray
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Finds the values in a lookup array at locations where array1 changes from NaN to True in array2.

    Parameters:
    -----------
        - array1: NumPy array (can contain NaN values).
        - array2: NumPy boolean array (should be of the same shape as array1).
        - lookup_array: NumPy array of the same shape as array1 and array2 to look up values.

    Returns:
    --------
        - Tuple containing:
        - Array of values from lookup_array at the identified locations.
        - Indices tuple of arrays representing the indices where the change occurs.
    """
    # Ensure the arrays have the same shape
    if not (array1.shape == array2.shape == lookup_array.shape):
        raise ValueError("All input arrays must have the same shape.")

    # Mask where array1 has NaN values
    nan_mask = np.isnan(array1)

    # Mask where array2 is True
    true_mask = array2.astype(bool)  # Ensure array2 is boolean

    # Find locations where array1 is NaN and array2 is True
    change_mask = nan_mask & true_mask

    # Get indices of these locations
    indices = np.where(change_mask)

    # Use the indices to look up values in the lookup_array
    values = lookup_array[indices]

    return values, indices


def has_overlapping_non_nan(stack: np.ndarray) -> np.bool:
    """
    Check if a stack of 2D arrays has any overlapping non-NaN values.

    Parameters:
    -----------
        - stack (np.ndarray): A 3D NumPy array where each "layer" is a 2D array.

    Returns:
    --------
        - bool: True if there are overlapping non-NaN values, False otherwise.
    """
    if stack.ndim != 3:
        raise ValueError("Input must be a 3D array (stack of 2D arrays).")

    # Create a mask where values are not NaN
    non_nan_mask = ~np.isnan(stack)

    # Sum the mask along the stacking axis (axis=0)
    overlap_count = np.sum(non_nan_mask, axis=0)

    # Check if any position has overlap (count > 1)
    return np.any(overlap_count > 1)


def common_true_locations(stack: np.ndarray) -> np.bool:
    """
    Check if any two 2D arrays in a 3D stack have overlapping `True` values.
    Treats NaN pixels as False.

    Parameters:
    -----------
        stack (np.ndarray): A 3D boolean array (a stack of 2D boolean arrays).

    Returns:
    --------
        bool: True if any two 2D arrays in the stack have overlapping `True` values,
          otherwise False.
    """
    if stack.ndim != 3:
        raise ValueError("Input must be a 3D stack of 2D arrays.")

    # Treat NaN as False by masking them out
    stack = np.nan_to_num(stack, nan=False)

    # Sum the stack along the first axis (layer-wise summation)
    overlap_sum = np.sum(stack, axis=0)

    # Check if any position has a value > 1, indicating overlap
    return np.any(overlap_sum > 1)


def preprocess_remove_extra_dim(da: xr.DataArray) -> xr.DataArray:
    """
    Preprocess function to remove an extra 'band' dimension and add a placeholder time dimension.
    Assumes the extra dimension is named 'band'.
    """
    da = da.squeeze(dim="band", drop=True)

    # Add a placeholder time based on file index
    file_index = da.encoding.get("source", "file_0").split("/")[-1]
    placeholder_time = datetime(2000, 1, 1)
    return da.expand_dims(time=[placeholder_time])


def open_veg_multifile(veg_base_path: str) -> xr.Dataset:
    """Open a multifile VegTransition output directory

    Correct timestamps must be applied after opening.

    Parameters:
    -----------
        - veg_base_path (string): Path to VegTransition output dir.

    Returns:
    --------
        (xr.Dataset) with time dimension.
    """
    dates = os.listdir(veg_base_path)
    dates = [
        i for i in dates if not re.search("[a-zA-Z]", i)
    ]  # rm any str with letters, which will handle the "run-metadata" dir
    dates = pd.to_datetime(dates)

    ds = xr.open_mfdataset(
        f"{veg_base_path}/**/*VEGTYPE.tif",
        preprocess=preprocess_remove_extra_dim,
        combine="nested",
        concat_dim="time",
    )

    ds["time"] = dates
    ds["band_data"] = ds["band_data"].astype(int)
    return ds


def pixel_sums_full_domain(ds: xr.Dataset) -> pd.DataFrame:
    """Process VegTransition output into timeseries .csv.

    Runtime for this function is 3 minutes or more for 25 year scenarios.
    """
    # Define unique vegetation types to analyze
    unique_values = [15, 16, 17, 18, 19, 20, 21, 22, 23, 26]

    # Use Xarray to count occurrences of each unique value
    # Create a DataArray mask for all unique values at once
    mask = xr.concat(
        [(ds["veg_type"] == value) for value in unique_values], dim="value"
    )
    # Assign unique values as a coordinate to the new dimension
    mask = mask.assign_coords(value=("value", unique_values))
    # Sum across spatial dimensions (y, x) to get counts for each value over time
    counts = mask.sum(dim=("y", "x"))

    df = counts.to_dataframe(name="count").reset_index()
    df = df.pivot(index="time", columns="value", values="count")
    return df


def wpu_sums(ds_veg: xr.Dataset, zones: xr.DataArray) -> pd.DataFrame:
    """
    Calculate timeseries of WPU pixel counts for vegetation data.

    This function computes a crosstabulation between vegetation types and zone identifiers
    across multiple timesteps, aggregating the results into a DataFrame. Note that NaN will
    be introduced into the output dataframe if a vegetation type exists in one timestep, but
    not others. This is equivalent to zero, and can be update to such if necessary.

    Parameters
    ----------
    ds_veg : xr.Dataset
        The vegetation dataset containing a `band_data` variable. This variable is a
        3D array with dimensions `(time, y, x)` representing vegetation types at
        each timestep.
    zones : xr.DataArray
        A 2D array with dimensions `(y, x)` representing zone identifiers (e.g., WPU zones).
        Must match the spatial dimensions of `band_data`.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns representing vegetation types and rows for each
        combination of zone and timestep. The `timestep` column indicates the time
        of each observation.
    """
    df_list = []

    for t in ds_veg.time.values:
        veg_2d = ds_veg["veg_type"].sel(time=t)

        # Convert the dask-backed DataArrays to NumPy
        veg_2d_np = veg_2d.compute()
        zones_np = zones.compute()

        df = crosstab(zones=zones_np, values=veg_2d_np)
        df.insert(loc=0, column="timestep", value=t)
        df_list.append(df)

    df_out = pd.concat(df_list)
    df_out.rename(columns={"zone": "wpu"}, inplace=True)
    df_out["wpu"] = df_out["wpu"].astype(int)
    # df_out.drop(columns=0, inplace=True)

    return df_out


def generate_filename(
    params: dict, parameter: str = None, base_path: str = None
) -> Path:
    """
    Generate a filename based on the Atchafalaya Master Plan (AMP) file naming convention.
    Missing parameters are skipped.

    Parameters:
    -----------
    params : dict
        Dictionary containing optional keys:
        - model : str
        - scenario : str
        - group : str
        - wpu : str
        - io_type : str
        - time_freq : str
        - year_range : str
        - output_version : str

        This dict is created in `VegTransition.step` and includes metadata from the
        current timestep as well as the model config file. It excludes "parameter",
        which is specified separately.

    parameter : str
        The name of the variable being saved, i.e. "VEGTYPE".

    base_path : str or Path, optional
        Base directory path where the file should be located.

    Returns:
    --------
    Path
        A `Path` object representing the full path to the generated file.
    """
    # Define the order of keys
    key_order = [
        "model",
        "water_year",
        "sea_level_condition",
        "flow_scenario",
        "year_range",
        "time_freq",
        "group",
        "wpu",
        "io_type",
        "output_version",
    ]

    # Collect available values from params in order
    values = [
        (
            params[key].upper()
            if isinstance(params.get(key), str)
            else params.get(key)
        )
        for key in key_order
        if key in params and params[key] is not None
    ]

    # Append parameter to the filename if provided
    if parameter:
        values.append(parameter)

    # Construct the filename
    filename = "AMP_" + "_".join(map(str, values))

    # Combine with base path if provided
    return Path(base_path) / filename if base_path else Path(filename)


def get_contiguous_regions(
    boolean_arr: np.ndarray, size_threshold: int
) -> np.ndarray:
    """Convenience function for calculating if a pixel is a
    member of a contiguous region of vegetation types.

    Parameters
    ------------
    boolean_arr : np.ndarray
        Boolean arr where True is the veg type to check for connectivity.

    size_threshold : int
        Number of connected pixels to consider a passing group size.

    Returns
    --------
    pixels belonging to large enough regions: np.ndarray
    """
    # step 1: label connected components
    structure = np.ones((3, 3), dtype=int)  # use 8-connectivity
    labeled_array, num_features = ndimage.label(
        boolean_arr, structure=structure
    )

    # step 2: count pixels per label
    label_sizes = np.bincount(labeled_array.ravel())

    # step 3: find large enough labels (excluding background label 0)
    large_labels = np.where(label_sizes[1:] >= size_threshold)[0] + 1

    # step 4: mask pixels belonging to large enough regions
    final_mask = np.isin(labeled_array, large_labels)

    return final_mask


def reduce_arr_by_mode(categorical_array: np.ndarray):
    """Reduce 60m array by mode to get 480m aggregation. Used for
    cases where mean is not appropriate (categorical vars).

    Params
    ------
    categorical_array : np.ndarray
        An array with integer categories.

    Return
    ------
    coarse_da : np.ndarray
        Array of input coarsed to 480m size, using mode to reduce pixels.
    """

    def mode_reduce(arr, axis):
        m = stats.mode(arr, axis=axis)
        return m[0]

    # define dims explicitly
    da = xr.DataArray(categorical_array, dims=["y", "x"])
    # coarsen to 8x8 blocks, applying mode function
    coarse_da = da.coarsen(y=8, x=8, boundary="pad").reduce(mode_reduce)
    return coarse_da


# def generate_filename(params: dict, parameter: str, base_path: str = None) -> Path:
#     """
#     Generate a filename based on the Atchafalaya Master Plan (AMP) file naming convention.

#     Parameters:
#     -----------
#     params : dict
#         Dictionary containing the following keys:
#         - model : str
#         - scenario : str
#         - group : str
#         - wpu : str
#         - io_type : str
#         - time_frame : str
#         - year_range : str
#         This dict is created in `VegTransition.step` and includes metadata from the
#         current timestep as well as the model config file. It excludes, "parameter"
#         wich is a required arg, and specified only when this function is called so
#         that the same timestep params dict can be used for different output
#         "parameters".

#     parameter : str
#         The name of the variable being saved, i.e. "VEGTYPE".

#     base_path : str or Path, optional
#         Base directory path where the file should be located.

#     Returns:
#     --------
#     Path
#         A `Path` object representing the full path to the generated file.
#     """
#     # Ensure keys are provided in the dictionary
#     required_keys = [
#         "model",
#         "scenario",
#         "group",
#         "wpu",
#         "io_type",
#         "time_freq",
#         "year_range",
#     ]
#     for key in required_keys:
#         if key not in params:
#             raise ValueError(f"Missing required key: '{key}' in params dictionary")

#     # Extract and process the values
#     model = params["model"].upper()
#     scenario = params["scenario"]
#     group = params["group"]
#     wpu = params["wpu"]
#     io_type = params["io_type"].upper()
#     time_freq = params["time_freq"].upper()
#     year_range = params["year_range"]

#     # file_extension = params["file_extension"].lower()

#     # Construct the filename
#     filename = f"AMP_{model}_{scenario}_{group}_{wpu}_{io_type}_{time_freq}_{year_range}_{parameter}"

#     # Combine with base path if provided
#     if base_path:
#         return Path(base_path) / filename
#     else:
#         return Path(filename)


def qc_tree_establishment_bool(
    water_depth: xr.Dataset,
) -> np.ndarray:
    """
    JV: Display Areas where establishment condition is met
    """
    mar_june = [3, 4, 5, 6]

    # Condition 1: MAR, APR, MAY, OR JUNE inundation depth <= 0
    filtered = water_depth.sel(
        time=water_depth["time"].dt.month.isin(mar_june)
    )
    tree_establish_bool = (filtered["height"] <= 0).any(dim="time").to_numpy()
    return tree_establish_bool


def qc_tree_establishment_info(
    water_depth: xr.Dataset,
) -> list[np.ndarray]:
    """
    JV: Depth in m for each month.
    """
    # .isel() required due to the month selection resulting in a single timestep 3D
    # dataarray, while the output format requires a 2D numpy array.
    march = (
        water_depth["height"]
        .sel(time=water_depth["time"].dt.month == 3)
        .isel(time=0)
    )
    april = (
        water_depth["height"]
        .sel(time=water_depth["time"].dt.month == 4)
        .isel(time=0)
    )
    may = (
        water_depth["height"]
        .sel(time=water_depth["time"].dt.month == 5)
        .isel(time=0)
    )
    june = (
        water_depth["height"]
        .sel(time=water_depth["time"].dt.month == 6)
        .isel(time=0)
    )
    return [
        march.to_numpy(),
        april.to_numpy(),
        may.to_numpy(),
        june.to_numpy(),
    ]


def qc_growing_season_inundation(
    water_depth: xr.Dataset,
) -> np.ndarray:
    """
    JV: Percentage of time flooded during the period from April 1 through September 30
    """
    gs = [4, 5, 6, 7, 8, 9]
    filtered = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
    pct_gs_inundation = (filtered["height"] > 0).mean(dim="time")
    return pct_gs_inundation.to_numpy()


def qc_growing_season_depth(
    water_depth: xr.Dataset,
) -> np.ndarray:
    """
    JV: Average water-depth during the period from April 1 through September 30
    """
    gs = [4, 5, 6, 7, 8, 9]
    filtered = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
    gs_depth = filtered["height"].mean(dim="time")
    return gs_depth.to_numpy()


def qc_annual_inundation_duration(
    water_depth: xr.Dataset,
) -> np.ndarray:
    """
    JV: Percentage of time flooded over the year
    """
    pct = (water_depth["height"] > 0).mean(dim="time")
    return pct.to_numpy()


def qc_annual_inundation_depth(
    water_depth: xr.Dataset,
) -> np.ndarray:
    """
    JV: Average water depth over the year.
    """
    mean_depth = water_depth["height"].mean(dim="time")
    return mean_depth.to_numpy()


def qc_annual_mean_salinity(salinity: np.ndarray) -> np.ndarray:
    """
    WARN: habitat-based salinity is equivalent to mean annual salinity
    as habitat only changes with yearly veg type update. Keeping this
    func as a placeholder for when salinity is a model variable.
    """
    return salinity


def dataset_attrs_to_df(ds: xr.Dataset, selected_attrs: list) -> pd.DataFrame:
    """
    Convert an xarray Dataset's variables and their attributes into a pandas DataFrame.
    Only attributes whose keys are present in the supplied list (selected_attrs) are included.

    Parameters:
    -----------
        ds (xr.Dataset): The input xarray Dataset.
        selected_attrs (list of str): A list of attribute keys to include (e.g., ['units', 'description']).

    Returns:
    ---------
        pd.DataFrame: A DataFrame with a 'variable' column for the variable name and additional
                      columns for each attribute in selected_attrs. If a variable does not have a given
                      attribute, the value will be None.
    """
    rows = []
    for var_name, var in ds.variables.items():
        row = {"variable": var_name}
        for key in selected_attrs:
            row[key] = var.attrs.get(key, None)
        rows.append(row)
    return pd.DataFrame(rows)


def load_sequence_csvs(directory: str) -> dict[int, int]:
    """load all csvs in a directory that include 'sequence' in the name

    keys in the output dict are the first part of each filename (before the first underscore)

    :param (str) directory: path to the directory containing sequence csvs
    :return (dict[int, int]): dictionary of "Year" and "Adjusted" as ints
    """
    csv_dict = {}

    for file in os.listdir(directory):
        if file.endswith(".csv") and "sequence" in file:
            prefix = file.split("_")[0]
            file_path = os.path.join(directory, file)
            csv_dict[prefix] = pd.read_csv(file_path)
            csv_dict[prefix] = csv_dict[prefix][["Adjusted", "Water Year"]]

    # concat dfs
    df = pd.concat([csv_dict["dry"], csv_dict["wet"]], axis=0)
    # convert to dict
    df.index = df["Water Year"]
    df.drop(columns="Water Year", inplace=True)
    dict_out = df["Adjusted"].to_dict()
    return dict_out


# def load_netcdf_variable_definitions(instance, filename: str) -> dict:
#     with open(filename, "r") as f:
#         raw_definitions = yaml.safe_load(f)

#     def resolve_path(obj, path_str):
#         # recursively get nested attributes
#         for attr in path_str.split("."):
#             obj = getattr(obj, attr)
#         return obj

#     variable_defs = {}
#     for varname, props in raw_definitions.items():
#         array_ref = resolve_path(instance, props["path"])
#         dtype = getattr(np, props["dtype"])
#         attrs = props["attrs"]
#         variable_defs[varname] = [array_ref, dtype, attrs]

#     return variable_defs
