import numpy as np
import functools


def qc_output(func):
    """
    Decorator to check if the output array differs from the input array.
    Logs a message if no changes are detected.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Expecting the veg_type array to be the second argument (args[1])
        veg_type_input = args[1].copy()  # Copy the input array to compare later

        # Run the wrapped function
        result = func(*args, **kwargs)

        # Compare the result with the input
        if np.array_equal(veg_type_input, result):
            logger = args[0]  # Assuming the first argument is the logger
            logger.info(f"No changes made to vegetation pixels in {func.__name__}.")
        return result

    return wrapper


def check_mask_overlap(masks):
    """Checks if independent masks have overlapping True pixels."""
    # Stack arrays and test for overlap
    qc_stacked = np.stack(masks)

    if np.logical_and.reduce(qc_stacked).any():
        logger.warning(
            "Valid transition pixels have overlap, indicating"
            "that some pixels are passing for both veg types"
            "but should be either. Check inputs."
        )


def find_nan_to_true_values(array1, array2, lookup_array):
    """
    Finds the values in a lookup array at locations where array1 changes from NaN to True in array2.

    Parameters:
    - array1: NumPy array (can contain NaN values)
    - array2: NumPy boolean array (should be of the same shape as array1)
    - lookup_array: NumPy array of the same shape as array1 and array2 to look up values

    Returns:
    - values: Array of values from lookup_array at the identified locations
    - indices: Tuple of arrays representing the indices where the change occurs
    """
    # Ensure the arrays have the same shape
    if not (array1.shape == array2.shape == lookup_array.shape):
        raise ValueError("All input arrays must have the same shape.")

    # Mask where array1 has NaN values
    nan_mask = np.isnan(array1)

    # Mask where array2 is True
    true_mask = array2.astype(bool)  # Ensure array2 is boolean

    # Find locations where array1 is NaN and array2 is True
    change_mask = nan_mask & true_mask

    # Get indices of these locations
    indices = np.where(change_mask)

    # Use the indices to look up values in the lookup_array
    values = lookup_array[indices]

    return values


def has_overlapping_non_nan(stack):
    """
    Check if a stack of 2D arrays has any overlapping non-NaN values.

    Parameters:
    - stack (np.ndarray): A 3D NumPy array where each "layer" is a 2D array.

    Returns:
    - bool: True if there are overlapping non-NaN values, False otherwise.
    """
    if stack.ndim != 3:
        raise ValueError("Input must be a 3D array (stack of 2D arrays).")

    # Create a mask where values are not NaN
    non_nan_mask = ~np.isnan(stack)

    # Sum the mask along the stacking axis (axis=0)
    overlap_count = np.sum(non_nan_mask, axis=0)

    # Check if any position has overlap (count > 1)
    return np.any(overlap_count > 1)
