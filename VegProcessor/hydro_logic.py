import numpy as np
import xarray as xr


def model_based_salinity():
    return NotImplementedError


def habitat_based_salinity(veg_type: np.ndarray | xr.DataArray) -> np.ndarray:
    """Get salinity defaults based on habitat type. If supplied a numpy array
    (as by VegTransition) the numpy array is returned with the same dims. If
    supplied an xr.DataArray, the same logic is applied, by the returned DataArray
    is downsampled from 60m to 480m using a spatial average (as in HSI).


    From Jenneke:

    If 60m pixel is saline marsh then set salinity to 18;
    Else if pixel is brackish marsh then set salinity to 8;
    Else if pixel is intermediate marsh then set salinity to 3.5;
    Else set salinity to 1.

    TODO check units

    Params:
        - veg_type (np.ndarray | xr.DataArray): array of current vegetation types.
    Returns:
        - np.ndarray: Salinty array with default values, for use
            when no salinity data is available.
    """
    if isinstance(veg_type, xr.DataArray):
        # Create salinity array initialized with default value of 1
        da = xr.full_like(veg_type, 1.0, dtype=float)

        # Apply conditions
        da = da.where(veg_type != 23, 18)
        da = da.where(veg_type != 22, 8)
        da = da.where(veg_type != 21, 3.5)

        # NaN is added (pad) for non-exact downscaling dims
        da = da.coarsen(x=8, y=8, boundary="pad").mean()
        return da

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


# def downsampled_habitat_based_salinity(veg_type: xr.DataArray) -> xr.DataArray:
#     """
#     Get salinity defaults based on habitat type for an xarray.DataArray. Logic
#     is the same as `hydro_logic.habitat_based_salnity`, while this function
#     operates on xr.DataArray object for easier and more robust resampling.

#     Params:
#         - veg_type (xr.DataArray): DataArray of current vegetation types.
#     Returns:
#         - xr.DataArray: Salinity DataArray with default values, for use
#             when no salinity data is available.
#     """
#     # Create salinity array initialized with default value of 1
#     salinity = xr.full_like(veg_type, 1.0, dtype=float)

#     # Apply conditions
#     salinity = salinity.where(veg_type != 23, 18)
#     salinity = salinity.where(veg_type != 22, 8)
#     salinity = salinity.where(veg_type != 21, 3.5)

#     return salinity

# def downsampled_habitat_based_salinity(veg_type: np.ndarray) -> np.ndarray:
#     """
#     Create habitat based salinity (60m), then downsample to 480m.
#     """
#     salinity = habitat_based_salinity(veg_type)

#     # DataArray with no geographic coordinates
#     da = xr.DataArray(
#         data=salinity,  # Input numpy array
#         dims=["x", "y"],  # Non-geographic dimensions
#         name="salinity",  # Optional: Name of the DataArray
#     )

#     da = da.coarsen(x=8, y=8, boundary="trim").mean()
