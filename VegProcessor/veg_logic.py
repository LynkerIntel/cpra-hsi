import numpy as np
import xarray as xr


def zone_v(veg_type: np.ndarray, depth: xr.Dataset) -> np.ndarray:
    """Calculate transition for Zone V

    MAR, APRIL, MAY, or JUNE
    inundation depth <= 0, AND
    GS inundation >20%

    Params:
        - zone_v (np.ndarray): array of current zone_v locations, with np.nan elsewhere
        - inundation (xr.Dataset): Dataset of 1 year of inundation depth from hydrologic model,
            created from water surface elevation and the domain DEM.

    Returns:
        - arr (np.ndarray): Output array of new values contining valid transitions,
            so this output array contains are mix of old zone v pixels and new values.
    """
    #return NotImplementedError

    # initialize return array with no changes
    veg_out = veg_type.copy()

    # subset to veg type v
    veg_v = veg_type[veg_type == int]

    # # create mask for depth condition
    mask_depth = depth.sel(slice=mar to june) <= 0

    # # create mask for temporal condition
    # inundation_gs = depth.sel(slice=growing season)
    # inundation_gs[inundation_gs != 0]

    # # create mask where both conditions are true
    # change_mask = # &

    # if change:
    #     arr[mask] = habitat int

    # return arr


def zone_iv(arg1, arg2, arg3) -> np.ndarray:
    """ """
    return NotImplementedError


def zone_iii(arg1, arg2, arg3) -> np.ndarray:
    """ """
    return NotImplementedError
