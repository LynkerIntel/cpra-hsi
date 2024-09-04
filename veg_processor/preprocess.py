import xarray as xr

# Define the coarsening factor (2x2 blocks)
coarsening_factor = {'x': 4, 'y': 4}

global_veg_type = 14
# Using global var here, rather than kwarg, due to bug in xarray `.reduce` 
# kwarg bug here: https://github.com/pydata/xarray/issues/8059

def count_vegtype_and_calculate_percentage(block: np.ndarray, axis: int) -> np.ndarray:
    """Get percentage cover of vegetation types in block group

    :param (np.ndarray) block: non-overlapping chunks from `.coarsen` function.
    :param (int) axis: used internally by `.reduce` to index the chunks
    :return (np.ndarray): coarsened chunk
    """
    # Sum over the provided axis
    count_ones = (block == global_veg_type).sum(axis=axis)
    total_cells = block.shape[axis[0]] * block.shape[axis[1]]
    return (count_ones / total_cells) * 100