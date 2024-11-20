import numpy as np


def inundation():
    """Get inundation based on DEM and WSE."""


def seasonal_inundation(named_season):
    """Get inundation for period (n months)

    arg should be named season OR months
    """
    return NotImplementedError

    if named_season == "gs":
        months = [5, 6, 7]
    elif named_season == "spring":
        months = [3, 4, 5]


# def model_based_salinity():


def habitat_based_salinity(veg_type: np.ndarray) -> np.ndarray:
    """Get salinity defaults based on habitat type.


    From Jenneke:

    If 60m pixel is saline marsh then set salinity to 18;
    Else if pixel is brackish marsh then set salinity to 8;
    Else if pixel is intermediate marsh then set salinity to 3.5;
    Else set salinity to 1.

    Params:
        - veg_type (np.ndarray): array of current vegetation types.
    Returns:
        - np.ndarray: Sailinty array with default values, for use
            when no salinity data is available.
    """
    # creat salinity, set all elements to 1
    salinity = np.ones_like(veg_type)

    # create masks
    saline_marsh_mask = veg_type == 23
    brackish_marsh_mask = veg_type == 22
    intermediate_marsh_mask = veg_type == 21

    salinity[saline_marsh_mask] = 18
    salinity[brackish_marsh_mask] = 8
    salinity[intermediate_marsh_mask] = 3.5

    return salinity
