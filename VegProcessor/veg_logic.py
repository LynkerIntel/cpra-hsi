import numpy as np
import xarray as xr
import datetime

import plotting


def zone_v(
    logger,
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    date: datetime.date,
    plot: bool = False,
) -> np.ndarray:
    """Calculate transition for Zone V

    MAR, APRIL, MAY, or JUNE
    inundation depth <= 0, AND
    GS inundation >20%

    Zone V = 15
    Zone IV = 16 (valid transition)

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
    # clone input for later plotting
    veg_type_input = veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone V (value 15)
    type_mask = veg_type == 15

    # Condition 1: MAR, APR, MAY inundation depth <= 0
    filtered_1 = water_depth.sel(time=slice(f"{date.year}-03", f"{date.year}-05"))
    condition_1 = (filtered_1["WSE_MEAN"] <= 0).any(dim="time")

    # Condition 2: Growing Season (GS) inundation > 20%
    filtered_2 = water_depth.sel(
        time=slice(growing_season["start"], growing_season["end"])
    )
    # Note: this assumes time is serially complete
    condition_2_pct = (filtered_2["WSE_MEAN"] > 0).mean(dim="time")
    condition_2 = condition_2_pct > 0.2

    transition_mask = np.logical_and(condition_1, condition_2)
    combined_mask = np.logical_and(type_mask, transition_mask)

    assert transition_mask.shape == combined_mask.shape == veg_type_input.shape

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
            np.where(condition_1, veg_type, np.nan),
            "Condition 1 (Inundation Depth <= 0)",
        )
        plotting.np_arr(
            np.where(condition_2, veg_type, np.nan),
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

    logger.info("Finished Zone V transitions.")

    # only return pixels that started as Zone V
    # by inverting the original mask
    veg_type[~type_mask] = np.nan
    return veg_type


def zone_iv(
    logger,
    veg_type: np.ndarray,
    water_depth: xr.Dataset,
    date: datetime.date,
    plot: bool = False,
) -> np.ndarray:
    """
    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation <20%
    Conditions: MAR, APR, MAY, or JUN inundation depth ≤ 0 cm AND GS Inundation ≥ 35%

    Zone IV: 16
    Zone III: 17
    """
    veg_type_input = veg_type.copy()
    growing_season = {"start": f"{date.year}-04", "end": f"{date.year}-09"}

    # Subset for veg type Zone IV (value 16)
    type_mask = veg_type == 16


def zone_iii(arg1, arg2, arg3) -> np.ndarray:
    """ """
    return NotImplementedError
