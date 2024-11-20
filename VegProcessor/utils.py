import xarray as xr
import numpy as np


# Function to create a new dataset based on a template
def create_dataset_from_template(template, new_variables):
    """
    Create an xarray.Dataset based on a template dataset.

    Parameters:
        - template (xr.Dataset): The template dataset.
        - new_variables (dict): Dictionary defining new variables.
          Keys are variable names, values are tuples (dims, data, attrs).

    Returns:
        - xr.Dataset: A new dataset based on the template.
    """
    # Copy dimensions and coordinates from the template
    dims = template.dims
    coords = {name: template.coords[name] for name in template.coords}

    # Create a new dataset
    new_ds = xr.Dataset(coords=coords)

    # Add new variables
    for var_name, (dims, data, attrs) in new_variables.items():
        new_ds[var_name] = xr.DataArray(data, dims=dims, attrs=attrs)

    # Optionally, copy global attributes from the template
    new_ds.attrs = template.attrs

    return new_ds


# # Define new variables to add
# new_variables = {
#     "precipitation": (("time", "lat", "lon"), np.random.rand(5, 4, 3), {"units": "mm/day"}),
#     "humidity": (("time", "lat", "lon"), np.random.rand(5, 4, 3), {"units": "%"}),
# }

# # Create the new dataset
# new_ds = create_dataset_from_template(template, new_variables)

# print(new_ds)
