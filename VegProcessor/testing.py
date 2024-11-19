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
