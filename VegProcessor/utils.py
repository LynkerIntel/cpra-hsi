import xarray as xr
import numpy as np
import pandas as pd


# Function to create a new dataset based on a template
def create_dataset_from_template(template, new_variables):
    """
    Create an xarray.Dataset based on a template dataset.

    Parameters:
        - template (xr.Dataset): The template dataset.
        - new_variables (dict): Dictionary defining new variables.
          Keys are variable names, values are tuples (data, attrs).
          - `data`: NumPy array of the same shape as the template's data variables.
          - `attrs`: Metadata for the variable.

    Returns:
        - xr.Dataset: A new dataset based on the template.
    """
    # Copy dimensions and coordinates from the template
    dims = template.dims
    coords = {name: template.coords[name] for name in template.coords}

    # Create a new dataset
    new_ds = xr.Dataset(coords=coords)

    # Validate and add new variables
    for var_name, (data, attrs) in new_variables.items():
        # Check that the shape matches the template
        if data.shape != template["WSE_MEAN"].shape:
            raise ValueError(
                f"Shape of variable '{var_name}' ({data.shape}) does not match "
                f"the template shape ({template['WSE_MEAN'].shape})."
            )
        # Add the variable
        new_ds[var_name] = xr.DataArray(
            data, dims=template["WSE_MEAN"].dims, attrs=attrs
        )

    # Optionally, copy global attributes from the template
    new_ds.attrs = template.attrs

    return new_ds


# # Define new variables to add (NumPy arrays must match the shape of the template)
# new_variables = {
#     "precipitation": (np.random.rand(5, 4, 3), {"units": "mm/day"}),
#     "humidity": (np.random.rand(5, 4, 3), {"units": "%"}),
# }

# # Create the new dataset
# new_ds = create_dataset_from_template(template, new_variables)

# print(new_ds)
