import numpy as np
import xarray as xr
import datetime
import os

import plotting
from testing import qc_output, find_nan_to_true_values

import matplotlib.pyplot as plt
import logging

# Configure the logger in VegTransition
logger = logging.getLogger("VegTransition")


@qc_output
def zone_v(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    date: datetime.date,
    plot: bool = False,
) -> np.ndarray:
    """Calculate transition for pixels starting in Zone V

    MAR, APRIL, MAY, or JUNE
    inundation depth <= 0, AND
    GS inundation >20%

    Zone V = 15
    Zone IV = 16 (valid transition)

    `to_numpy()` is used after xarray operations in order to stack and combined masks as
    np.ndarrays.

    Params:
        - logger: pass main logger to this function
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - date (datetime.date): Date to derive year for filtering.
        - plot (bool): If True, plots the array before and after transformation.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Zone V
    """
    veg_name = "Zone V"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = (
        timestep_output_dir + f"/figs/{veg_name.lower().replace(" ", "_")}/"
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone V (value 15)
    type_mask = veg_type == 15

    # these should be combined eventually
    veg_type = np.where(type_mask, veg_type, np.nan)
    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY, JUNE inundation depth <= 0
    filtered_1 = water_depth.sel(time=slice(f"{date.year}-03", f"{date.year}-06"))
    condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

    # Condition 2: Growing Season (GS) inundation > 20%
    filtered_2 = water_depth.sel(
        time=slice(growing_season["start"], growing_season["end"])
    )
    # Note: this assumes time is serially complete
    condition_2_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = (condition_2_pct > 0.2).to_numpy()

    stacked_masks = np.stack((type_mask, condition_1, condition_2))
    combined_mask = np.logical_and.reduce(stacked_masks)

    if np.logical_and(np.isnan(veg_type), combined_mask).any():
        logger.warning(
            "Logical error: Pixels that were masked by the veg_type mask "
            "have been identified as True in the combined mask. "
            "Check inputs."
        )

    # apply transition
    veg_type[combined_mask] = 16
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    if plot:
        # Plotting code should be careful to use
        # veg_type_input, when showing the input
        # array, and veg_type, when showing the
        # output array
        plotting.np_arr(
            arr=veg_type_input,
            title="Input - Zone V",
            outpath=timestep_output_dir,  # Explicit argument
        )
        plotting.np_arr(
            type_mask,
            "Veg Type Mask (Zone V)",
            outpath=timestep_output_dir,  # Explicit argument
        )
        plotting.np_arr(
            np.where(condition_1, veg_type_input, np.nan),
            "Condition 1 (Inundation Depth <= 0)",
            description,
            timestep_output_dir,  # Explicit argument
        )
        plotting.np_arr(
            np.where(condition_2, veg_type_input, np.nan),
            "Condition 2 (GS Inundation > 20%)",
            description,
            timestep_output_dir,  # Explicit argument
        )
        plotting.np_arr(
            np.where(combined_mask, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met)",
            description,
            timestep_output_dir,  # Explicit argument
        )
        plotting.np_arr(
            veg_type,
            "Output - Updated Veg Types",
            description,
            timestep_output_dir,  # Explicit argument
        )

    logger.info("Finished transitions with input type: Zone V")
    return veg_type


@qc_output
def zone_iv(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    date: datetime.date,
    plot: bool = False,
) -> np.ndarray:
    """Calculate transitions for pixels starting in Zone IV

    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation <20%
    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation ≥ 35%

    Zone IV: 16
    Zone III: 17

    Params:
        - logger: pass main logger to this function
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - date (datetime.date): Date to derive year for filtering.
        - plot (bool): If True, plots the array before and after transformation.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Zone IV
    """
    veg_name = "Zone IV"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = (
        timestep_output_dir + f"/figs/{veg_name.lower().replace(" ", "_")}/"
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone IV (value 16)
    type_mask = veg_type == 16
    # Set other veg types to nan
    # veg_type[~type_mask] = 999
    # veg_type_input[~type_mask] = 999
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY inundation depth <= 0
    filtered_1 = water_depth.sel(time=slice(f"{date.year}-03", f"{date.year}-06"))
    condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

    # Condition 2: Growing Season (GS) inundation < 20%
    filtered_2 = water_depth.sel(
        time=slice(growing_season["start"], growing_season["end"])
    )
    # get pct duration of inundation (i.e. depth > 0)
    # Note: this assumes time is serially complete
    condition_2_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = (condition_2_pct < 0.2).to_numpy()

    # Condition 3:  Growing Season (GS) inundation >= 35%
    condition_3_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_3 = (condition_3_pct >= 0.35).to_numpy()

    # get pixels that meet zone v criteria
    stacked_masks_v = np.stack((condition_1, condition_2))
    combined_mask_v = np.logical_and.reduce(stacked_masks_v)

    # get pixels that meet zone III criteria
    stacked_masks_iii = np.stack((~combined_mask_v, condition_1, condition_3))
    combined_mask_iii = np.logical_and.reduce(stacked_masks_iii)

    # ensure there is no overlap between
    # zone v and zone iii pixels
    if np.logical_and(combined_mask_v, combined_mask_iii).any():
        logger.warning(
            "Valid transition pixels have overlap, indicating"
            "that some pixels are passing for both veg types"
            "but should be either. Check inputs."
        )

    # update valid transition types
    veg_type[combined_mask_v] = 15
    veg_type[combined_mask_iii] = 17
    # reapply mask, because depth conditions don't include type.
    veg_type = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    logger.info("Output veg types: %s", np.unique(veg_type))

    if plot:
        plotting.np_arr(veg_type_input, "Input - Zone IV", outpath=timestep_output_dir)
        plotting.np_arr(
            type_mask, "Veg Type Mask (Zone IV)", outpath=timestep_output_dir
        )
        plotting.np_arr(
            np.where(condition_1, veg_type_input, np.nan),
            "Condition 1 (Inundation Depth <= 0)",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(condition_2, veg_type_input, np.nan),
            "Condition 2 (GS Inundation < 20%)",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(condition_3, veg_type_input, np.nan),
            "Condition 3 (GS Inundation > 35%)",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_v, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> V",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_iii, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> III",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            veg_type, "Output - Updated Veg Types", description, timestep_output_dir
        )

        logger.info("Finished transitions with input type: Zone IV")

    return veg_type


@qc_output
def zone_iii(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    date: datetime.date,
    plot: bool = False,
) -> np.ndarray:
    """Calculate transition for pixels starting in Zone III


    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation <15%
    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation ≥ 80%

    Zone IV: 16
    Zone III: 17
    Zone II: 18

    Params:
        - logger: pass main logger to this function
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - date (datetime.date): Date to derive year for filtering.
        - plot (bool): If True, plots the array before and after transformation.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Zone III
    """
    veg_name = "Zone III"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = (
        timestep_output_dir + f"/figs/{veg_name.lower().replace(" ", "_")}/"
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone III (value 17)
    type_mask = veg_type == 17
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY, JUNE inundation % TIME <= 0
    filtered_1 = water_depth.sel(time=slice(f"{date.year}-03", f"{date.year}-06"))
    condition_1_pct = (filtered_1["WSE_MEAN"] > 0).mean(dim="time")
    condition_1 = (condition_1_pct == 0).to_numpy()

    # Condition 2: Growing Season (GS) inundation < 15%
    filtered_2 = water_depth.sel(
        time=slice(growing_season["start"], growing_season["end"])
    )
    # get pct duration of inundation (i.e. depth > 0)
    # Note: this assumes time is serially complete
    condition_2_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = (condition_2_pct < 0.15).to_numpy()

    # Condition 3:  Growing Season (GS) inundation >= 80%
    condition_3_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_3 = (condition_3_pct >= 0.8).to_numpy()

    # get pixels that meet zone iv criteria
    stacked_masks_iv = np.stack((condition_1, condition_2))
    combined_mask_iv = np.logical_and.reduce(stacked_masks_iv)

    # get pixels that meet zone II criteria
    stacked_masks_ii = np.stack((~combined_mask_iv, condition_1, condition_3))
    combined_mask_ii = np.logical_and.reduce(stacked_masks_ii)

    # ensure there is no overlap between
    # zone v and zone iii pixels
    if np.logical_and(combined_mask_iv, combined_mask_ii).any():
        logger.warning(
            "Valid transition pixels have overlap, indicating"
            "that some pixels are passing for both veg types"
            "but should be either. Check inputs."
        )

    # update valid transition types
    veg_type[combined_mask_iv] = 16
    veg_type[combined_mask_ii] = 18
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    if plot:
        plotting.np_arr(veg_type_input, "Input - Zone III", outpath=timestep_output_dir)
        plotting.np_arr(
            type_mask, "Veg Type Mask (Zone III)", outpath=timestep_output_dir
        )
        plotting.np_arr(
            np.where(condition_1, veg_type_input, np.nan),
            "Condition 1 (Inundation Time == 0% )",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(condition_2, veg_type_input, np.nan),
            "Condition 2 (GS Inundation < 15%)",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(condition_3, veg_type_input, np.nan),
            "Condition 3 (GS Inundation > 80%)",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_iv, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> IV",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_ii, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> II",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            veg_type, "Output - Updated Veg Types", description, timestep_output_dir
        )

    logger.info("Finished transitions with input type: Zone III")
    return veg_type


@qc_output
def zone_ii(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    date: datetime.date,
    plot: bool = False,
) -> np.ndarray:
    """Calculate transition for pixels starting in Zone II


    Condition_1: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation <70% TIME
    Condition_2: GS Inundation < 20% TIME
    Condition_3: Annual inundation == 100% AND annual inundation depth > 10cm

    Zone IV: 16
    Zone III: 17
    Zone II: 18

    Params:
        - logger: pass main logger to this function
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - date (datetime.date): Date to derive year for filtering.
        - plot (bool): If True, plots the array before and after transformation.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Zone II
    """
    veg_name = "Zone II"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = (
        timestep_output_dir + f"/figs/{veg_name.lower().replace(" ", "_")}/"
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone II (value 18)
    type_mask = veg_type == 18
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY inundation depth <= 0
    filtered_1 = water_depth.sel(time=slice(f"{date.year}-03", f"{date.year}-06"))
    condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

    # Condition 2: Annual inundation < 70% TIME
    # Note: this assumes time is serially complete
    condition_2_pct = (water_depth["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = (condition_2_pct < 0.7).to_numpy()

    # Condition 3: Growing Season (GS) inundation < 20%
    filtered_3 = water_depth.sel(
        time=slice(growing_season["start"], growing_season["end"])
    )
    # get pct duration of inundation (i.e. depth > 0)
    # Note: this assumes time is serially complete
    condition_3_pct = (filtered_3["WSE_MEAN"] > 0).mean(dim="time")
    condition_3 = (condition_3_pct < 0.2).to_numpy()

    # Condition 4:  Annual inundation == 100%
    condition_4_pct = (water_depth["WSE_MEAN"] > 0).mean(dim="time")
    condition_4 = (condition_4_pct == 1).to_numpy()

    # Condition 5: Annual inundation depth <= 10cm #UNIT
    condition_5 = (water_depth["WSE_MEAN"] <= 0.1).all(dim="time").to_numpy()

    # get pixels that meet zone iii criteria
    stacked_masks_iii = np.stack((condition_1, condition_2))
    combined_mask_iii = np.logical_and.reduce(stacked_masks_iii)

    # get pixels that meet fresh shrub criteria
    stacked_masks_fresh_shrub = np.stack((~combined_mask_iii, condition_3))
    combined_mask_fresh_shrub = np.logical_and.reduce(stacked_masks_fresh_shrub)

    # get pixels that meet fresh marsh criteria
    stacked_masks_fresh_marsh = np.stack(
        (
            ~combined_mask_iii,
            ~combined_mask_fresh_shrub,
            condition_4,
            condition_5,
        )
    )
    combined_mask_fresh_marsh = np.logical_and.reduce(stacked_masks_fresh_marsh)

    # Stack arrays and test for overlap
    qc_stacked = np.stack(
        [
            combined_mask_iii,
            combined_mask_fresh_shrub,
            combined_mask_fresh_marsh,
        ]
    )
    if np.logical_and.reduce(qc_stacked).any():
        logger.warning(
            "Valid transition pixels have overlap, indicating"
            "that some pixels are passing for both veg types"
            "but should be either. Check inputs."
        )

    # update valid transition types
    veg_type[combined_mask_iii] = 17
    veg_type[combined_mask_fresh_shrub] = 19
    veg_type[combined_mask_fresh_marsh] = 20
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    if plot:
        plotting.np_arr(veg_type_input, "Input - Zone II", outpath=timestep_output_dir)
        plotting.np_arr(
            type_mask, "Veg Type Mask (Zone II)", outpath=timestep_output_dir
        )
        plotting.np_arr(
            np.where(condition_1, veg_type_input, np.nan),
            "Condition 1: inundation depth <= 0",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(condition_2, veg_type_input, np.nan),
            "Condition 2: Annual inundation < 70% TIME",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(condition_3, veg_type_input, np.nan),
            "Condition 3: Growing Season (GS) inundation < 20%",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_iii, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> III",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_fresh_shrub, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> Fresh Shrub",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_fresh_marsh, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> Fresh Marsh",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            veg_type, "Output - Updated Veg Types", description, timestep_output_dir
        )

    logger.info("Finished transitions with input type: Zone II")
    return veg_type


@qc_output
def fresh_shrub(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    date: datetime.date,
    plot: bool = False,
) -> np.ndarray:
    """Calculate transition for pixels starting as fresh shrub


    Condition_1: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm
    Condition_2: Annual Inundation >= 80% TIME
    Condition_3: Annual inundation > 40% TIME

    Zone IV: 16
    Zone III: 17
    Zone II: 18
    Fresh Shrub: 19


    Params:
        - logger: pass main logger to this function
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - date (datetime.date): Date to derive year for filtering.
        - plot (bool): If True, plots the array before and after transformation.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Fresh Shrub
    """
    veg_name = "Fresh Shrub"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = (
        timestep_output_dir + f"/figs/{veg_name.lower().replace(" ", "_")}/"
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone II (value 18)
    type_mask = veg_type == 19
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY inundation depth <= 0
    filtered_1 = water_depth.sel(time=slice(f"{date.year}-03", f"{date.year}-06"))
    condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

    # Condition 2: Annual inundation > 80% TIME
    # Note: this assumes time is serially complete
    condition_2_pct = (water_depth["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = (condition_2_pct > 0.8).to_numpy()

    # Condition 3: Growing Season (GS) inundation >= 40%
    filtered_3 = water_depth.sel(
        time=slice(growing_season["start"], growing_season["end"])
    )
    # get pct duration of inundation (i.e. depth > 0)
    # Note: this assumes time is serially complete
    condition_3_pct = (filtered_3["WSE_MEAN"] > 0).mean(dim="time")
    condition_3 = (condition_3_pct >= 0.4).to_numpy()

    # get pixels that meet zone ii criteria
    # Use logical AND to find locations where all arrays are True
    stacked_masks_ii = np.stack((condition_1, condition_2))
    combined_mask_ii = np.logical_and.reduce(stacked_masks_ii)

    # get pixels that meet fresh marsh criteria
    # inverse mask to ensure no overlap
    stacked_masks_fresh_marsh = np.stack((~combined_mask_ii, condition_3))
    combined_mask_fresh_marsh = np.logical_and.reduce(stacked_masks_fresh_marsh)

    # Stack arrays and test for overlap
    qc_stacked = np.stack(
        [
            combined_mask_ii,
            combined_mask_fresh_marsh,
        ]
    )
    if np.logical_and.reduce(qc_stacked).any():
        logger.warning(
            "Valid transition pixels have overlap, indicating"
            "that some pixels are passing for both veg types"
            "but should be either. Check inputs."
        )

    # update valid transition types
    veg_type[combined_mask_ii] = 18
    veg_type[combined_mask_fresh_marsh] = 20
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    if plot:
        plotting.np_arr(
            veg_type_input, "Input - Fresh Shrub", outpath=timestep_output_dir
        )
        plotting.np_arr(
            type_mask, "Veg Type Mask (Fresh Shrub)", outpath=timestep_output_dir
        )
        plotting.np_arr(
            np.where(condition_1, veg_type_input, np.nan),
            "Condition 1: inundation depth <= 0",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(condition_2, veg_type_input, np.nan),
            "Condition 2: Annual inundation >= 80% TIME",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(condition_3, veg_type_input, np.nan),
            "Condition 3: Growing Season (GS) inundation >= 40%",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_ii, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> Zone II",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_fresh_marsh, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> Fresh Marsh",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            veg_type, "Output - Updated Veg Types", description, timestep_output_dir
        )

    logger.info("Finished transitions with input type: Fresh Shrub")
    return veg_type


@qc_output
def fresh_marsh(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    salinity: np.ndarray,
    date: datetime.date,
    plot: bool = False,
) -> np.ndarray:
    """Calculate transition for pixels starting as Fresh Marsh


    Condition_1: GS Inundation == 100% TIME
    Condition_2: mean GS depth > 20cm
    Condition_3: mean ANNUAL salinity >= 2ppt
    Condition_4: APR:SEP inundation < 30% TIME
    Condition_5: MAR, APR, MAY, JUNE inundation <= 0
    Condition_6: ANNUAL inundation > 80% TIME

    Zone IV: 16
    Zone III: 17
    Zone II: 18
    Fresh Shrub: 19
    Fresh Marsh: 20


    Params:
        - logger: pass main logger to this function
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - salinity (np.ndarray): array of salinity, from HH model OR defaults.
        - date (datetime.date): Date to derive year for filtering.
        - plot (bool): If True, plots the array before and after transformation.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Fresh Marsh
    """
    veg_name = "Fresh Marsh"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = (
        timestep_output_dir + f"/figs/{veg_name.lower().replace(" ", "_")}/"
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Fresh Marsh (value 20)
    type_mask = veg_type == 20
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition_1: GS Inundation == 100% TIME
    filtered_1 = water_depth.sel(
        time=slice(growing_season["start"], growing_season["end"])
    )
    condition_1_pct = (filtered_1["WSE_MEAN"] > 0).mean(dim="time")
    condition_1 = (condition_1_pct == 1).to_numpy()

    # Condition_2: MEAN GS depth > 20cm
    condition_2 = (filtered_1["WSE_MEAN"].mean(dim="time") > 0.2).to_numpy()

    # Condition_3: mean ANNUAL salinity >= 2ppt
    # TODO: when monthly inputs are available, this will need
    # to accept monthly values for defaults and model output
    condition_3 = salinity >= 2

    # Condition_4: APR:SEP inundation < 30% TIME
    filtered_2 = water_depth.sel(time=slice(f"{date.year}-04", f"{date.year}-09"))
    condition_4_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_4 = (condition_4_pct >= 0.4).to_numpy()

    # Condition_5: MAR, APR, MAY, JUNE inundation <= 0
    filtered_3 = water_depth.sel(time=slice(f"{date.year}-03", f"{date.year}-06"))
    condition_5 = (filtered_3["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

    # Condition_6: ANNUAL inundation > 80% TIME
    condition_6_pct = (water_depth["WSE_MEAN"] > 0).mean(dim="time")
    condition_6 = (condition_6_pct > 0.8).to_numpy()

    # get pixels that meet Water criteria
    # Use logical AND to find locations where all arrays are True
    stacked_mask_water = np.stack((condition_1, condition_2))
    combined_mask_water = np.logical_and.reduce(stacked_mask_water)

    # get pixels that meet Intermediate Marsh criteria
    stacked_masks_intermediate_marsh = np.stack(
        (
            ~combined_mask_water,
            condition_3,
        )
    )
    combined_mask_intermediate_marsh = np.logical_and.reduce(
        stacked_masks_intermediate_marsh
    )

    # get pixels that meet Fresh Shrub criteria
    stacked_masks_fresh_shrub = np.stack(
        (
            ~combined_mask_water,
            ~combined_mask_intermediate_marsh,
            condition_4,
        )
    )
    combined_mask_fresh_shrub = np.logical_and.reduce(stacked_masks_fresh_shrub)

    # get pixels that meet Zone II criteria
    stacked_masks_zone_ii = np.stack(
        (
            ~combined_mask_water,
            ~combined_mask_intermediate_marsh,
            ~combined_mask_fresh_shrub,
            condition_5,
            condition_6,
        )
    )
    combined_mask_zone_ii = np.logical_and.reduce(stacked_masks_zone_ii)

    # update valid transition types
    veg_type[combined_mask_water] = 26
    veg_type[combined_mask_intermediate_marsh] = 21
    veg_type[combined_mask_fresh_shrub] = 19
    veg_type[combined_mask_zone_ii] = 18

    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    if plot:
        plotting.np_arr(
            veg_type_input, "Input - Fresh Marsh", outpath=timestep_output_dir
        )
        plotting.np_arr(
            type_mask, "Veg Type Mask (Fresh Marsh)", outpath=timestep_output_dir
        )
        plotting.np_arr(
            np.where(combined_mask_water, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> Water",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_intermediate_marsh, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> Intermediate Marsh",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_fresh_shrub, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> Fresh Shrub",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            np.where(combined_mask_zone_ii, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> Zone II",
            description,
            timestep_output_dir,
        )
        plotting.np_arr(
            veg_type, "Output - Updated Veg Types", description, timestep_output_dir
        )

    logger.info("Finished transitions with input type: Fresh Marsh")
    return veg_type


# @qc_output
# def intermediate_marsh(
#     veg_type: np.ndarray,
#     water_depth: xr.Dataset,
#     date: datetime.date,
#     plot: bool = False,
# ) -> np.ndarray:
#     """Calculate transition for pixels starting in Intermediate Marsh

#     Condition_1: GS Inundation > 80% TIME
#     Condition_2: Average ANNUAL salinity >= 5ppt
#     Condition_3: Average ANNUAL salinity < 1 ppt


#     Zone IV: 16
#     Zone III: 17
#     Zone II: 18
#     Intermediate Marsh: 21

#     Params:
#         - logger: pass main logger to this function
#         - veg_type (np.ndarray): array of current vegetation types.
#         - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
#             created from water surface elevation and the domain DEM.
#         - date (datetime.date): Date to derive year for filtering.
#         - plot (bool): If True, plots the array before and after transformation.

#     Returns:
#         - np.ndarray: Modified vegetation type array with updated transitions
#             for pixels starting as Intermediate Marsh
#     """
#     logger.info("Starting transitions with input type: Intermediate Marsh")
#     description = "Input Veg Type: Intermediate Marsh"
#     # clone input
#     veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
#     growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

#     # Subset for veg type Intermediate Marsh (value 21)
#     type_mask = veg_type == 21
#     veg_type = np.where(type_mask, veg_type, np.nan)
#     veg_type_input = np.where(type_mask, veg_type, np.nan)

#     nan_count = np.sum(np.isnan(veg_type))
#     logger.info("Input NaN count: %d", nan_count)


#     # Condition 1: Annual inundation > 80% TIME
#     filtered_1 = water_depth.sel(
#         time=slice(growing_season["start"], growing_season["end"])
#     )
#     condition_1_pct = (filtered_1["WSE_MEAN"] > 0).mean(dim="time")
#     condition_1 = (condition_1_pct > 0.8).to_numpy()


#     # Condition_2: Average ANNUAL salinity >= 5ppt


#     # Condition 1: MAR, APR, MAY inundation depth <= 0
#     filtered_1 = water_depth.sel(time=slice(f"{date.year}-03", f"{date.year}-06"))
#     condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

#     # Condition 2: Annual inundation < 70% TIME
#     # Note: this assumes time is serially complete
#     condition_2_pct = (water_depth["WSE_MEAN"] > 0).mean(dim="time")
#     condition_2 = (condition_2_pct < 0.7).to_numpy()

#     # Condition 3: Growing Season (GS) inundation < 20%
#     filtered_3 = water_depth.sel(
#         time=slice(growing_season["start"], growing_season["end"])
#     )
#     # get pct duration of inundation (i.e. depth > 0)
#     # Note: this assumes time is serially complete
#     condition_3_pct = (filtered_3["WSE_MEAN"] > 0).mean(dim="time")
#     condition_3 = (condition_3_pct < 0.2).to_numpy()

#     # Condition 4:  Annual inundation == 100%
#     condition_4_pct = (water_depth["WSE_MEAN"] > 0).mean(dim="time")
#     condition_4 = (condition_4_pct == 1).to_numpy()

#     # Condition 5: Annual inundation depth <= 10cm #UNIT
#     condition_5 = (water_depth["WSE_MEAN"] <= 0.1).all(dim="time").to_numpy()

#     # get pixels that meet zone iii criteria
#     stacked_masks_iii = np.stack((condition_1, condition_2))
#     combined_mask_iii = np.logical_and.reduce(stacked_masks_iii)

#     # get pixels that meet fresh shrub criteria
#     stacked_masks_fresh_shrub = np.stack((~combined_mask_iii, condition_3))
#     combined_mask_fresh_shrub = np.logical_and.reduce(stacked_masks_fresh_shrub)

#     # get pixels that meet fresh marsh criteria
#     stacked_masks_fresh_marsh = np.stack(
#         (
#             ~combined_mask_iii,
#             ~combined_mask_fresh_shrub,
#             condition_4,
#             condition_5,
#         )
#     )
#     combined_mask_fresh_marsh = np.logical_and.reduce(stacked_masks_fresh_marsh)

#     # Stack arrays and test for overlap
#     qc_stacked = np.stack(
#         [
#             combined_mask_iii,
#             combined_mask_fresh_shrub,
#             combined_mask_fresh_marsh,
#         ]
#     )
#     if np.logical_and.reduce(qc_stacked).any():
#         logger.warning(
#             "Valid transition pixels have overlap, indicating"
#             "that some pixels are passing for both veg types"
#             "but should be either. Check inputs."
#         )

#     # update valid transition types
#     veg_type[combined_mask_iii] = 17
#     veg_type[combined_mask_fresh_shrub] = 19
#     veg_type[combined_mask_fresh_marsh] = 20
#     # reapply mask, because depth conditions don't include type
#     veg_type = np.where(type_mask, veg_type, np.nan)

#     logger.info("Output veg types: %s", np.unique(veg_type))

#     nan_count = np.sum(np.isnan(veg_type))
#     logger.info("Output NaN count: %d", nan_count)

#     if plot:
#         # plotting code should be careful to use
#         # veg_type_input, when showing the input
#         # array, and veg_type, when showing the
#         # output array
#         plotting.np_arr(
#             veg_type_input,
#             "Input - Zone II",
#             # description,
#         )
#         plotting.np_arr(
#             type_mask,
#             "Veg Type Mask (Zone II)",
#             # description,
#         )
#         plotting.np_arr(
#             np.where(condition_1, veg_type_input, np.nan),
#             "Condition 1: inundation depth <= 0",
#             description,
#         )
#         plotting.np_arr(
#             np.where(condition_2, veg_type_input, np.nan),
#             "Condition 2: Annual inundation < 70% TIME",
#             description,
#         )
#         plotting.np_arr(
#             np.where(condition_3, veg_type_input, np.nan),
#             "Condition 3: Growing Season (GS) inundation < 20%",
#             description,
#         )
#         plotting.np_arr(
#             np.where(combined_mask_iii, veg_type_input, np.nan),
#             "Combined Mask (All Conditions Met) -> III",
#             description,
#         )
#         plotting.np_arr(
#             np.where(combined_mask_fresh_shrub, veg_type_input, np.nan),
#             "Combined Mask (All Conditions Met) -> Fresh Shrub",
#             description,
#         )
#         plotting.np_arr(
#             np.where(combined_mask_fresh_marsh, veg_type_input, np.nan),
#             "Combined Mask (All Conditions Met) -> Fresh Marsh",
#             description,
#         )
#         plotting.np_arr(
#             veg_type,
#             "Output - Updated Veg Types",
#             description,
#         )

#     logger.info("Finished transitions with input type: Zone II")
#     return veg_type
