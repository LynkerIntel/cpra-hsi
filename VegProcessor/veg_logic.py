import numpy as np
import xarray as xr
import datetime
import os

import plotting
from testing import qc_output, find_nan_to_true_values

import matplotlib.pyplot as plt
import logging

import testing

# Configure the logger in VegTransition
logger = logging.getLogger("VegTransition")


# @qc_output
def zone_v(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
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
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Zone V
    """
    veg_name = "Zone V"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Zone V (value 15)
    type_mask = veg_type == 15

    # these should be combined eventually
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)
    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY, OR JUNE inundation depth <= 0
    filtered_1 = water_depth.sel(time=water_depth["time"].dt.month.isin(mar_june))
    condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

    # Condition 2: Growing Season (GS) inundation > 20%
    filtered_2 = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
    # Note: this assumes time is serially complete
    condition_2_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = (condition_2_pct > 0.2).to_numpy()

    stacked_masks = np.stack((condition_1, condition_2))
    combined_mask = np.logical_and.reduce(stacked_masks)

    # apply transition
    veg_type[combined_mask] = 16
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid WSE mask
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    plotting.np_arr(
        type_mask,
        "Veg Type Mask (Zone V)",
        out_path=timestep_output_dir,
    )
    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone V Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Zone V")
    return veg_type


# @qc_output
def zone_iv(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
) -> np.ndarray:
    """Calculate transitions for pixels starting in Zone IV

    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation <20%
    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation ≥ 35%

    Zone IV: 16
    Zone III: 17

    Params:
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Zone IV
    """
    veg_name = "Zone IV"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Zone IV (value 16)
    type_mask = veg_type == 16

    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY, JUNE inundation depth <= 0 (using OR)
    filtered_1 = water_depth.sel(time=water_depth["time"].dt.month.isin(mar_june))
    condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

    # Condition 2: Growing Season (GS) inundation < 20%
    filtered_2 = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
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

    # Stack arrays and test for overlap
    qc_stacked = np.stack(
        [
            combined_mask_v,
            combined_mask_iii,
        ]
    )

    if testing.common_true_locations(qc_stacked):
        raise ValueError("Stacked arrays cannot have overlapping True pixels.")

    # update valid transition types
    veg_type[combined_mask_v] = 15
    veg_type[combined_mask_iii] = 17
    # reapply mask, because depth conditions don't include type.
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid WSE mask
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    logger.info("Output veg types: %s", np.unique(veg_type))

    # plotting.np_arr(veg_type_input, "Input - Zone IV", out_path=timestep_output_dir)
    plotting.np_arr(type_mask, "Veg Type Mask (Zone IV)", out_path=timestep_output_dir)

    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone IV Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Zone IV")

    return veg_type


# @qc_output
def zone_iii(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
) -> np.ndarray:
    """Calculate transition for pixels starting in Zone III


    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation <15%
    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation ≥ 80%

    Zone IV: 16
    Zone III: 17
    Zone II: 18

    Params:
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Zone III
    """
    veg_name = "Zone III"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Zone III (value 17)
    type_mask = veg_type == 17
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY, JUNE inundation % TIME <= 0%
    filtered_1 = water_depth.sel(time=water_depth["time"].dt.month.isin(mar_june))
    condition_1 = (filtered_1["WSE_MEAN"] == 0).any(dim="time")
    # condition_1 = (condition_1_pct == 0).to_numpy()

    # Condition 2: Growing Season (GS) inundation < 15%
    filtered_2 = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
    # get pct duration of inundation (i.e. depth > 0)
    # Note: this assumes time is serially complete
    condition_2_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = (condition_2_pct < 0.15).to_numpy()

    # Condition 3:  ANNUAL inundation >= 80%
    condition_3_pct = (water_depth["WSE_MEAN"] > 0).mean(dim="time")
    condition_3 = (condition_3_pct >= 0.8).to_numpy()

    # get pixels that meet zone iv criteria
    stacked_masks_iv = np.stack((condition_1, condition_2))
    combined_mask_iv = np.logical_and.reduce(stacked_masks_iv)

    # get pixels that meet zone II criteria
    stacked_masks_ii = np.stack((~combined_mask_iv, condition_1, condition_3))
    combined_mask_ii = np.logical_and.reduce(stacked_masks_ii)

    # Stack arrays and test for overlap
    qc_stacked = np.stack(
        [
            combined_mask_iv,
            combined_mask_ii,
        ]
    )

    if testing.common_true_locations(qc_stacked):
        raise ValueError("Stacked arrays cannot have overlapping True pixels.")

    # update valid transition types
    veg_type[combined_mask_iv] = 16
    veg_type[combined_mask_ii] = 18
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid WSE mask
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    # plotting.np_arr(veg_type_input, "Input - Zone III", out_path=timestep_output_dir)
    plotting.np_arr(type_mask, "Veg Type Mask (Zone III)", out_path=timestep_output_dir)

    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone III Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Zone III")
    return veg_type


# @qc_output
def zone_ii(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
) -> np.ndarray:
    """Calculate transition for pixels starting in Zone II


    Condition_1: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation <70% TIME
    Condition_2: GS Inundation < 20% TIME
    Condition_3: Annual inundation == 100% AND annual inundation depth > 10cm

    Zone IV: 16
    Zone III: 17
    Zone II: 18

    Params:
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Zone II
    """
    veg_name = "Zone II"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"
    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Zone II (value 18)
    type_mask = veg_type == 18
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY inundation depth <= 0
    filtered_1 = water_depth.sel(time=water_depth["time"].dt.month.isin(mar_june))
    condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

    # Condition 2: Annual inundation < 70% TIME
    # Note: this assumes time is serially complete
    condition_2_pct = (water_depth["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = (condition_2_pct < 0.7).to_numpy()

    # Condition 3: Growing Season (GS) inundation < 20%
    filtered_3 = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
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

    if testing.common_true_locations(qc_stacked):
        raise ValueError("Stacked arrays cannot have overlapping True pixels.")

    # update valid transition types
    veg_type[combined_mask_iii] = 17
    veg_type[combined_mask_fresh_shrub] = 19
    veg_type[combined_mask_fresh_marsh] = 20
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid model locations
    # this prevents pixels where water_depth is NaN from
    # being included in results. This is more simple than adding
    # specific nan handling to each condition, but has the downside
    # that intermediate plots (i.e. arrays during the condition building)
    # will not be a reliable source of transition info during debugging.
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    # plotting.np_arr(veg_type_input, "Input - Zone II", out_path=timestep_output_dir)
    plotting.np_arr(type_mask, "Veg Type Mask (Zone II)", out_path=timestep_output_dir)

    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone II Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Zone II")
    return veg_type


# @qc_output
def fresh_shrub(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
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
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Fresh Shrub
    """
    veg_name = "Fresh Shrub"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"

    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Zone II (value 18)
    type_mask = veg_type == 19
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: MAR, APR, MAY inundation depth <= 0
    filtered_1 = water_depth.sel(time=water_depth["time"].dt.month.isin(mar_june))
    condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time").to_numpy()

    # Condition 2: Annual inundation > 80% TIME
    # Note: this assumes time is serially complete
    condition_2_pct = (water_depth["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = (condition_2_pct > 0.8).to_numpy()

    # Condition 3: Growing Season (GS) inundation >= 40%
    filtered_3 = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
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
    if testing.common_true_locations(qc_stacked):
        raise ValueError("Stacked arrays cannot have overlapping True pixels.")

    # update valid transition types
    veg_type[combined_mask_ii] = 18
    veg_type[combined_mask_fresh_marsh] = 20
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid WSE mask
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    # plotting.np_arr(veg_type_input, "Input - Fresh Shrub", out_path=timestep_output_dir)
    plotting.np_arr(
        type_mask, "Veg Type Mask (Fresh Shrub)", out_path=timestep_output_dir
    )
    # plotting.np_arr(
    #     veg_type, "Output - Updated Veg Types", description, timestep_output_dir
    # )
    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone Fresh Shrub Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Fresh Shrub")
    return veg_type


# @qc_output
def fresh_marsh(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    salinity: np.ndarray,
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
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.
        - salinity (np.ndarray): array of salinity for WY (either from model output of defaults)

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Fresh Marsh
    """
    veg_name = "Fresh Marsh"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"

    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    apr_sep = [4, 5, 6, 7, 8, 9]
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Fresh Marsh (value 20)
    type_mask = veg_type == 20
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition_1: GS Inundation == 100% TIME
    filtered_1 = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
    condition_1_pct = (filtered_1["WSE_MEAN"] > 0).mean(dim="time")
    condition_1 = (condition_1_pct == 1).to_numpy()

    # Condition_2: MEAN GS depth > 20cm
    condition_2 = (filtered_1["WSE_MEAN"].mean(dim="time") > 0.2).to_numpy()

    # Condition_3: mean ANNUAL salinity >= 2ppt
    # TODO: when monthly inputs are available, this will need
    # to accept monthly values for defaults and model output
    condition_3 = salinity >= 2

    # Condition_4: APR:SEP inundation < 30% TIME
    filtered_2 = water_depth.sel(time=water_depth["time"].dt.month.isin(apr_sep))
    condition_4_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_4 = (condition_4_pct < 0.3).to_numpy()

    # Condition_5: MAR, APR, MAY, JUNE inundation <= 0
    filtered_3 = water_depth.sel(time=water_depth["time"].dt.month.isin(mar_june))
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

    # Stack arrays and test for overlap
    qc_stacked = np.stack(
        [
            combined_mask_water,
            combined_mask_intermediate_marsh,
            combined_mask_fresh_shrub,
        ]
    )

    if testing.common_true_locations(qc_stacked):
        raise ValueError("Stacked arrays cannot have overlapping True pixels.")

    # update valid transition types
    veg_type[combined_mask_water] = 26
    veg_type[combined_mask_intermediate_marsh] = 21
    veg_type[combined_mask_fresh_shrub] = 19
    veg_type[combined_mask_zone_ii] = 18

    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid WSE mask
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    # plotting.np_arr(veg_type_input, "Input - Fresh Marsh", out_path=timestep_output_dir)
    plotting.np_arr(
        type_mask, "Veg Type Mask (Fresh Marsh)", out_path=timestep_output_dir
    )
    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone Fresh Marsh Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Fresh Marsh")
    return veg_type


# @qc_output
def intermediate_marsh(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    salinity: np.ndarray,
) -> np.ndarray:
    """Calculate transition for pixels starting in Intermediate Marsh

    Condition_1: GS Inundation > 80% TIME
    Condition_2: Average ANNUAL salinity >= 5ppt
    Condition_3: Average ANNUAL salinity < 1 ppt


    Zone IV: 16
    Zone III: 17
    Zone II: 18
    Intermediate Marsh: 21

    Params:
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.
        - salinity (np.ndarray): array of salinity for WY (either from model output of defaults)

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Intermediate Marsh
    """
    veg_name = "Intermediate Marsh"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"

    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Intermediate Marsh (value 21)
    type_mask = veg_type == 21
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: Growind Season inundation > 80% TIME
    filtered_1 = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
    condition_1_pct = (filtered_1["WSE_MEAN"] > 0).mean(dim="time")
    condition_1 = (condition_1_pct > 0.8).to_numpy()

    # Condition_2: Average ANNUAL salinity >= 5ppt
    # TODO: when monthly inputs are available, this will need
    # to accept monthly values for defaults and model output
    condition_2 = salinity >= 5

    # Condition_3: Average ANNUAL salinity < 1ppt
    # TODO: when monthly inputs are available, this will need
    # to accept monthly values for defaults and model output
    condition_3 = salinity < 1

    # get pixels that meet water criteria
    # (reassigning for consistency with other vars)
    combined_mask_water = condition_1

    # get pixels that meet brackish marsh criteria
    stacked_mask_brackish_marsh = np.stack(
        (
            ~combined_mask_water,
            condition_2,
        )
    )
    combined_mask_brackish_marsh = np.logical_and.reduce(stacked_mask_brackish_marsh)

    # get pixels that meet fresh marsh criteria
    stacked_mask_fresh_marsh = np.stack(
        (
            ~combined_mask_water,
            ~combined_mask_brackish_marsh,
            condition_3,
        )
    )
    combined_mask_fresh_marsh = np.logical_and.reduce(stacked_mask_fresh_marsh)

    # Stack arrays and test for overlap
    qc_stacked = np.stack(
        [
            combined_mask_water,
            combined_mask_brackish_marsh,
            combined_mask_fresh_marsh,
        ]
    )

    if testing.common_true_locations(qc_stacked):
        raise ValueError("Stacked arrays cannot have overlapping True pixels.")

    # update valid transition types
    veg_type[combined_mask_water] = 26
    veg_type[combined_mask_brackish_marsh] = 22
    veg_type[combined_mask_fresh_marsh] = 20
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid WSE mask
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    plotting.np_arr(
        type_mask,
        f"Veg Type Mask ({veg_name})",
        # description,
    )
    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone Intermediate Marsh Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Intermediate Marsh")
    return veg_type


# @qc_output
def brackish_marsh(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    salinity: np.ndarray,
) -> np.ndarray:
    """Calculate transition for pixels starting in Brackish Marsh

    Condition_1: GS Inundation > 80% TIME
    Condition_2: Average ANNUAL salinity >= 12ppt
    Condition_3: Average ANNUAL salinity < 4 ppt


    Zone IV: 16
    Zone III: 17
    Zone II: 18
    Intermediate Marsh: 21
    Brackish Marsh: 22

    Params:
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.
        - salinity (np.ndarray): array of salinity for WY (either from model output or defaults)

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Brackish Marsh
    """
    veg_name = "Brackish Marsh"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"

    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Brackish Marsh (value 22)
    type_mask = veg_type == 22
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: GS inundation > 80% TIME
    filtered_1 = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
    condition_1_pct = (filtered_1["WSE_MEAN"] > 0).mean(dim="time")
    condition_1 = (condition_1_pct > 0.8).to_numpy()

    # Condition_2: Average ANNUAL salinity >= 5ppt
    # TODO: when monthly inputs are available, this will need
    # to accept monthly values for defaults and model output
    condition_2 = salinity >= 12

    # Condition_3: Average ANNUAL salinity < 1ppt
    # TODO: when monthly inputs are available, this will need
    # to accept monthly values for defaults and model output
    condition_3 = salinity < 4

    # get pixels that meet water criteria
    # (reassigning for consistency with other vars)
    combined_mask_water = condition_1

    # get pixels that meet saline marsh criteria
    stacked_mask_saline_marsh = np.stack(
        (
            ~combined_mask_water,
            condition_2,
        )
    )
    combined_mask_saline_marsh = np.logical_and.reduce(stacked_mask_saline_marsh)

    # get pixels that meet intermediate marsh criteria
    stacked_mask_intermediate_marsh = np.stack(
        (
            ~combined_mask_water,
            ~combined_mask_saline_marsh,
            condition_3,
        )
    )
    combined_mask_intermediate_marsh = np.logical_and.reduce(
        stacked_mask_intermediate_marsh
    )

    # Stack arrays and test for overlap
    qc_stacked = np.stack(
        [
            combined_mask_water,
            combined_mask_saline_marsh,
            combined_mask_intermediate_marsh,
        ]
    )

    if testing.common_true_locations(qc_stacked):
        raise ValueError("Stacked arrays cannot have overlapping True pixels.")

    # update valid transition types
    veg_type[combined_mask_water] = 26
    veg_type[combined_mask_saline_marsh] = 23
    veg_type[combined_mask_intermediate_marsh] = 21
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid WSE mask
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    plotting.np_arr(
        type_mask,
        f"Veg Type Mask ({veg_name})",
        # description,
    )
    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone Brackish Marsh Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Brackish Marsh")
    return veg_type


# @qc_output
def saline_marsh(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    salinity: np.ndarray,
) -> np.ndarray:
    """Calculate transition for pixels starting in Saline Marsh

    Condition_1: GS Inundation > 80% TIME
    Condition_2: Average ANNUAL salinity < 11ppt

    Intermediate Marsh: 21
    Brackish Marsh: 22
    Saline Marsh: 23

    Params:
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.
        - salinity (np.ndarray): array of salinity for WY (either from model output or defaults)

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Saline Marsh
    """
    veg_name = "Saline Marsh"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"

    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Saline Marsh (value 23)
    type_mask = veg_type == 23
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition 1: GS inundation > 80% TIME
    filtered_1 = water_depth.sel(time=water_depth["time"].dt.month.isin(gs))
    condition_1_pct = (filtered_1["WSE_MEAN"] > 0).mean(dim="time")
    condition_1 = (condition_1_pct > 0.8).to_numpy()

    # Condition_2: Average ANNUAL salinity < 11ppt
    # TODO: when monthly inputs are available, this will need
    # to accept monthly values for defaults and model output
    condition_2 = salinity < 11

    # get pixels that meet water criteria
    # (reassigning for consistency with other vars)
    combined_mask_water = condition_1

    # get pixels that meet brackish marsh criteria
    stacked_mask_brackish_marsh = np.stack(
        (
            ~combined_mask_water,
            condition_2,
        )
    )
    combined_mask_brackish_marsh = np.logical_and.reduce(stacked_mask_brackish_marsh)

    # Stack arrays and test for overlap
    qc_stacked = np.stack(
        [
            combined_mask_water,
            combined_mask_brackish_marsh,
        ]
    )

    if testing.common_true_locations(qc_stacked):
        raise ValueError("Stacked arrays cannot have overlapping True pixels.")

    # update valid transition types
    veg_type[combined_mask_water] = 26
    veg_type[combined_mask_brackish_marsh] = 22
    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid WSE mask
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))
    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    plotting.np_arr(
        type_mask,
        f"Veg Type Mask ({veg_name})",
        # description,
    )
    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone Saline Marsh Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Saline Marsh")
    return veg_type


# @qc_output
def water(
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    timestep_output_dir: str,
    salinity: np.ndarray,
) -> np.ndarray:
    """Calculate transition for pixels starting in Water

    Condition_1: Average ANNUAL depth < 5cm
    Condition_2: Average ANNUAL salinity < 2
    Condition_3: Average ANNUAL depth < 5cm
    Condition_4: Average ANNUAL salinity < 5ppt
    Condition_5: Average ANNUAL depth < 5cm
    Condition_6: Average ANNUAL salinity < 12ppt
    Condition_7: Average ANNUAL depth < 5cm

    Intermediate Marsh: 21
    Brackish Marsh: 22
    Saline Marsh: 23
    Water: 26

    Params:
        - veg_type (np.ndarray): array of current vegetation types.
        - water_depth (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.
        - timestep_output_dir (str): location for output raster data or plots.
        - salinity (np.ndarray): array of salinity for WY (either from model output or defaults)

    Returns:
        - np.ndarray: Modified vegetation type array with updated transitions
            for pixels starting as Water
    """
    veg_name = "Water"
    logger.info("Starting transitions with input type: %s", veg_name)
    description = f"Input Veg Type: {veg_name}"

    timestep_output_dir = os.path.join(
        timestep_output_dir, veg_name.lower().replace(" ", "_")
    )
    os.makedirs(timestep_output_dir, exist_ok=True)

    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    mar_june = [3, 4, 5, 6]
    gs = [4, 5, 6, 7, 8, 9]

    # Subset for veg type Water (value 26)
    type_mask = veg_type == 26
    veg_type = np.where(type_mask, veg_type, np.nan)
    veg_type_input = np.where(type_mask, veg_type, np.nan)

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Input NaN count: %d", nan_count)

    # Condition_1: Average ANNUAL depth < 5cm (and Condition 3 & 5)
    condition_1_3_5_7 = water_depth["WSE_MEAN"].mean(dim="time")
    condition_1_3_5_7 = (condition_1_3_5_7 < 0.05).to_numpy()

    # Condition_2: Average Salinity < 2
    condition_2 = salinity < 2

    # Condition_4: Average ANNUAL salinity < 5ppt
    condition_4 = salinity < 5

    # Condition_6: Average ANNUAL salinity < 12ppt
    condition_6 = salinity < 12

    # get pixels that meet fresh marsh criteria
    stacked_mask_fresh_marsh = np.stack(
        (
            condition_1_3_5_7,
            condition_2,
        )
    )
    combined_mask_fresh_marsh = np.logical_and.reduce(stacked_mask_fresh_marsh)

    # get pixels that meet intermediate marsh criteria
    stacked_mask_intermediate_marsh = np.stack(
        (
            ~combined_mask_fresh_marsh,
            condition_1_3_5_7,
            condition_4,
        )
    )
    combined_mask_intermediate_marsh = np.logical_and.reduce(
        stacked_mask_intermediate_marsh
    )

    # get pixels that meet brackish marsh criteria
    stacked_mask_brackish_marsh = np.stack(
        (
            ~combined_mask_fresh_marsh,
            ~combined_mask_intermediate_marsh,
            condition_1_3_5_7,
            condition_6,
        )
    )
    combined_mask_brackish_marsh = np.logical_and.reduce(stacked_mask_brackish_marsh)

    # get pixels that meet Saline marsh criteria
    stacked_mask_saline_marsh = np.stack(
        (
            ~combined_mask_fresh_marsh,
            ~combined_mask_intermediate_marsh,
            ~combined_mask_brackish_marsh,
            condition_1_3_5_7,
        )
    )
    combined_mask_saline_marsh = np.logical_and.reduce(stacked_mask_saline_marsh)

    # Stack arrays and test for overlap
    qc_stacked = np.stack(
        [
            combined_mask_fresh_marsh,
            combined_mask_intermediate_marsh,
            combined_mask_brackish_marsh,
            combined_mask_saline_marsh,
        ]
    )

    if testing.common_true_locations(qc_stacked):
        raise ValueError("Stacked arrays cannot have overlapping True pixels.")

    # update valid transition types
    veg_type[combined_mask_fresh_marsh] = 20
    veg_type[combined_mask_intermediate_marsh] = 21
    veg_type[combined_mask_brackish_marsh] = 22
    veg_type[combined_mask_saline_marsh] = 23

    # reapply mask, because depth conditions don't include type
    veg_type = np.where(type_mask, veg_type, np.nan)
    # apply valid WSE mask
    valid_wse = water_depth["WSE_MEAN"][0].notnull().values
    veg_type = np.where(valid_wse, veg_type, np.nan)

    logger.info("Output veg types: %s", np.unique(veg_type))

    nan_count = np.sum(np.isnan(veg_type))
    logger.info("Output NaN count: %d", nan_count)

    plotting.np_arr(
        type_mask,
        f"Veg Type Mask ({veg_name})",
        # description,
    )
    plotting.sum_changes(
        veg_type_input,
        veg_type,
        plot_title="Zone Water Sum Changes",
        out_path=timestep_output_dir,
    )

    logger.info("Finished transitions with input type: Water")
    return veg_type
