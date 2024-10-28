import numpy as np


def zone_v(arg1, arg2, arg3) -> np.ndarray:
    """ """
    return NotImplementedError


def zone_iv(arg1, arg2, arg3) -> np.ndarray:
    """ """
    return NotImplementedError


def zone_iii(arg1, arg2, arg3) -> np.ndarray:
    """ """
    return NotImplementedError


def habitat_based_salinity() -> np.ndarray:
    """Get salinity defaults based on habitat type.


    From Jenneke:

    If 60m pixel is saline marsh then set salinity to 18;
    Else if pixel is brackish marsh then set salinity to 8;
    Else if pixel is intermediate marsh then set salinity to 3.5;
    Else set salinity to 1.
    """
    return NotImplementedError

    # condition 1
    self.veg_type["saline march"] == 18

    # mask_1 = self.veg_type == "saline marsh"
    # si_1[mask_1] = ((4.5 * self.v1_pct_open_water[mask_1]) / 100) + 0.1
