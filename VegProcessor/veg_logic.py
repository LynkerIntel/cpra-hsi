import numpy as np
import xarray as xr
import datetime

import plotting
from testing import qc_output


@qc_output
def zone_v(
    logger,
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
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
    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone V (value 15)
    type_mask = veg_type == 15

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

    # apply transition
    veg_type[combined_mask] = 16

    if plot:
        # this might be cleaned up or simplified.
        # plotting code should be careful to use
        # veg_type_input, when showing the input
        # array, and veg_type, when showing the
        # output array
        plotting.np_arr(veg_type_input, "Input - Zone V")
        plotting.np_arr(
            type_mask,
            "Veg Type Mask (Zone V)",
        )
        plotting.np_arr(
            np.where(condition_1, veg_type_input, np.nan),
            "Condition 1 (Inundation Depth <= 0)",
        )
        plotting.np_arr(
            np.where(condition_2, veg_type_input, np.nan),
            "Condition 2 (GS Inundation > 20%)",
        )
        plotting.np_arr(
            np.where(combined_mask, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met)",
        )
        plotting.np_arr(
            veg_type,
            "Output - Updated Veg Types",
        )

    logger.info("Finished transitions with start type: Zone V")

    # only return pixels that started as Zone V
    # by inverting the original mask
    veg_type[~type_mask] = np.nan
    return veg_type


@qc_output
def zone_iv(
    logger,
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
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
    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone IV (value 16)
    type_mask = veg_type == 16

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
    stacked_masks_v = np.stack((type_mask, condition_1, condition_2))
    combined_mask_v = np.logical_and.reduce(stacked_masks_v)

    # get pixels that meet zone III criteria
    stacked_masks_iii = np.stack((type_mask, condition_1, condition_3))
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

    if plot:
        # plotting code should be careful to use
        # veg_type_input, when showing the input
        # array, and veg_type, when showing the
        # output array
        plotting.np_arr(veg_type_input, "Input - Zone IV")
        plotting.np_arr(
            type_mask,
            "Veg Type Mask (Zone IV)",
        )
        plotting.np_arr(
            np.where(condition_1, veg_type_input, np.nan),
            "Condition 1 (Inundation Depth <= 0)",
        )
        plotting.np_arr(
            np.where(condition_2, veg_type_input, np.nan),
            "Condition 2 (GS Inundation < 20%)",
        )
        plotting.np_arr(
            np.where(condition_3, veg_type_input, np.nan),
            "Condition 3 (GS Inundation > 35%)",
        )
        plotting.np_arr(
            np.where(combined_mask_v, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> V",
        )
        plotting.np_arr(
            np.where(combined_mask_iii, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> III",
        )
        plotting.np_arr(
            veg_type,
            "Output - Updated Veg Types",
        )

    logger.info("Finished transitions with start type: Zone IV")
    return veg_type


@qc_output
def zone_iii(
    logger,
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    date: datetime.date,
    plot: bool = False,
) -> np.ndarray:
    """Calculate transition for pixels starting in Zone III


    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation <20%
    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation ≥ 35%

    Zone IV: 16
    Zone III: 17
    """
    # clone input
    veg_type, veg_type_input = veg_type.copy(), veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone IV (value 16)
    type_mask = veg_type == 16

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
    stacked_masks_v = np.stack((type_mask, condition_1, condition_2))
    combined_mask_v = np.logical_and.reduce(stacked_masks_v)

    # get pixels that meet zone III criteria
    stacked_masks_iii = np.stack((type_mask, condition_1, condition_3))
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

    if plot:
        # plotting code should be careful to use
        # veg_type_input, when showing the input
        # array, and veg_type, when showing the
        # output array
        plotting.np_arr(veg_type_input, "Input - Zone IV")
        plotting.np_arr(
            type_mask,
            "Veg Type Mask (Zone IV)",
        )
        plotting.np_arr(
            np.where(condition_1, veg_type_input, np.nan),
            "Condition 1 (Inundation Depth <= 0)",
        )
        plotting.np_arr(
            np.where(condition_2, veg_type_input, np.nan),
            "Condition 2 (GS Inundation < 20%)",
        )
        plotting.np_arr(
            np.where(condition_3, veg_type_input, np.nan),
            "Condition 3 (GS Inundation > 35%)",
        )
        plotting.np_arr(
            np.where(combined_mask_v, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> V",
        )
        plotting.np_arr(
            np.where(combined_mask_iii, veg_type_input, np.nan),
            "Combined Mask (All Conditions Met) -> III",
        )
        plotting.np_arr(
            veg_type,
            "Output - Updated Veg Types",
        )

    logger.info("Finished transitions with start type: Zone IV")
    return veg_type
