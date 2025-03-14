import numpy as np
import xarray as xr


def model_based_salinity():
    return NotImplementedError


def habitat_based_salinity(
        veg_type: np.ndarray | xr.DataArray, 
        domain : np.ndarray, 
        cell : bool = False
    ) -> np.ndarray:
    """
    Get salinity defaults based on habitat type. Can be supplied a numpy array or
    xr.DataArray. A np.ndarray is always returned.

    From Jenneke:

    If 60m pixel is saline marsh then set salinity to 18;
    Else if pixel is brackish marsh then set salinity to 8;
    Else if pixel is intermediate marsh then set salinity to 3.5;
    Else set salinity to 1.

    TODO properly implement 12 month salinity (static as of now)

    Params:
    --------
    veg_type : (np.ndarray | xr.DataArray) 
        array of current vegetation types.
    cell : bool
        True if output should be downsampled to 480m grid cell size.
    domain : np.ndarray
        Domain to mask output array by.
    
    Returns
    --------
    salinity : np.ndarray
        Salinty array with default values, for use when no salinity 
        data is available.
    """
    if not isinstance(veg_type, xr.DataArray):
        veg_type = xr.DataArray(veg_type)

    salinity = xr.full_like(veg_type, 1.0, dtype=float)

    salinity = xr.where(veg_type == 23, 18, salinity)
    salinity = xr.where(veg_type == 22, 8, salinity)
    salinity = xr.where(veg_type == 21, 3.5, salinity)

    # mask to veg domain
    salinity = salinity.where(domain, np.nan)

    if cell:
        salinity = salinity.coarsen(x=8, y=8, boundary="pad").mean()
    
    return salinity.to_numpy()


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
