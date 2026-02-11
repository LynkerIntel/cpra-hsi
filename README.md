# CPRA HSI & Vegetation Models
___
This repo contains code for the Habitat Suitability Modeling (HSI) task, and the vegetation transition model, which serves as a partial basis for the HSI data input.

___
### `VegProcessor/`
Description: This folder contains code for processing inputs and executing the vegetation transition model, as well as the subsequent HSI model implementation. It handles input data preprocessing, vegetation type transitions based on environmental conditions (e.g., water depth, salinity), and generating outputs for analysis and visualization. The `VegTransition` model is designed to simulate vegetation dynamics over time and provide inputs for Habitat Suitability Index (HSI) models.

#### Contents:
- `veg_transition.py`: Framework for vegetation transition modeling, implementing rules and conditions for vegetation type changes over time. The `VegTransition` class is initialized with a `config.yaml` which defines the model parameters. The `run()` method executes the model.
- `hsi.py`: Framework for running the HSI models over the domain. `HSI` is a child class of `VegTransition` and inherits much of it's functionality for updating state variables over time.
- `batch_run.py`: Script for running multiple VegTransition and HSI scenarios in batch mode with automatic validation of outputs.
- `veg_logic.py`: Detailed implementation of vegetation transition rules, handling specific conditions and constraints for various vegetation types.
- `test.py`: Unit testing of vegetation zone logic.
- `utils.py`: General utility functions for working with file paths, datasets, and common logic used throughout the model. Includes functions to generate the 25-year sequenves. Also includes runtime testing that occurs during execution.
- `plotting.py`: Tools for visualizing input data, transition results, and model outputs.
- `run.ipynb`: Example workflow demonstrating how to execute the vegetation model, HSI models, and post processing of results.

___
#### `VegProcessor/configs`

- Configuration files for defining a vegetation transition model run.

___
#### `VegProcessor/sequences`
- CSV files defining the ordering of analog years with the 25 year sequences.

___
#### `VegProcessor/species_hsi`
Description: this folder contains the individual species HSI logic.
#### Contents:
- `alligator.py`
- `baldeagle.py`
- `bass.py`
- `blackbear.py`
- `blackcrappie.py`
- `blhwva.py`
- `bluecrab.py`
- `catfish.py`
- `crawfish.py`
- `gizzardshad.py`
- `swampwva.py`

___
#### `scripts/`
Description: Utility scripts for data preprocessing and conversion.
#### Contents:
- `nc_to_zarr.py`: Converts raw hydrologic NetCDF files to Zarr format, handling CRS normalization and optional reprojection to a reference grid.

___
### Setup for Model Runs & Development

#### **1. Clone the Repository**

Ensure **Git** is installed, then open a terminal and run:

```bash
git clone https://github.com/LynkerIntel/cpra-hsi.git
cd cpra-hsi
```

---

#### **2. Create and Activate a Conda Environment**

An environment file, `environment_multiplatform.yml`, is provided. Create the Conda environment using:

```bash
conda env create -f environment/environment_multiplatform.yml
```

Activate the environment:

```bash
conda activate cpra_env
```

This installs the necessary dependencies.

The environment is currently not defined with pinned versions, in order to maximize capatibility. Conda will install the latest version of packages that does not create conflicts. This will change to pinned versions.


---

#### **3. Configure the Model**

In order to run scenarios (i.e. base or sea level rise), config files must be created for each desired run, for both `VegTransition` and `HSI`.

`post_process()` methods are included for both classes. These may evolve over time, but are generally for (1) reducing model output to only necessary vars, periods, or locations, and (2) summarizing the results.

- `VegProcessor/veg_config_**`: Specifies vegetation transition model settings, raster data paths, and output locations.
- `VegProcessor/hsi_config_**`: Defines parameters for running the Habitat Suitability Index model.


#### **4. Preprocess Hydrologic Input Data (NetCDF to Zarr)**

Raw hydrologic model outputs (from HEC-RAS, MIKE, or Delft3D) must be converted to Zarr format before running the models. The `scripts/nc_to_zarr.py` script handles this conversion and performs necessary preprocessing:

- Converts NetCDF files to Zarr stores for faster I/O
- Normalizes CRS metadata across different hydrologic model formats
- Renames spatial dimensions to standard `y`/`x` naming
- Optionally reprojects data to match a reference raster grid

**Basic usage:**

```bash
# Convert all .nc files in a directory to .zarr stores
python scripts/nc_to_zarr.py /path/to/hydro/netcdf_files/

# Specify a custom output directory
python scripts/nc_to_zarr.py /path/to/hydro/netcdf_files/ -o /path/to/zarr_output/

# Set time dimension chunk size (default: 1)
python scripts/nc_to_zarr.py /path/to/hydro/netcdf_files/ --time-chunks 10

# Reproject to match a reference raster (e.g., DEM grid)
python scripts/nc_to_zarr.py /path/to/hydro/netcdf_files/ --match-raster /path/to/dem.tif
```

**Example workflow:**

```bash
# Convert HEC-RAS stage outputs, reprojecting to match the 60m DEM
python scripts/nc_to_zarr.py /data/hydro/hec_stage/ --match-raster /data/rasters/dem_60m.tif

# Convert MIKE salinity outputs
python scripts/nc_to_zarr.py /data/hydro/mike_salinity/ -o /data/zarr/mike_salinity_zarr/
```

By default, output Zarr stores are written to a sibling directory named `{input_dir}_zarr`. For example, converting `/data/hydro/hec_stage/` produces `/data/hydro/hec_stage_zarr/`.

After conversion, update your config files to point to the `.zarr` stores in the `netcdf_hydro_path` and `netcdf_salinity_path` fields.

> **Note on NetCDF Input Files:**
>
> If you prefer to use NetCDF directly (without Zarr conversion), HDF5-based NetCDF4
> files are required for xarray parallel operations (NetCDF3 classic is not compatible).
> This terminal command will batch convert files:
> ```bash
> for f in *.nc; do nccopy -k 4 -d 4 "$f" "netcdf4_$f"; done
> ```

---

## Running Models

There are two ways to run the models:
1. **Interactive (Notebook/Script)**: Run individual scenarios using Python
2. **Batch Mode**: Run multiple scenarios sequentially using `batch_run.py`

### Option 1: Interactive Execution

To execute the vegetation transition model:

```python
from VegProcessor.veg_transition import VegTransition

# Initialize the model with a config file
Veg = VegTransition(config_file="./configs/veg_config.yaml")

# Run the model
Veg.run()
Veg.post_process()  # optionally produce summaries
```

Keep in mind that the HSI models depend on the `VegTransition` output, and must always be executed second. To run the **Habitat Suitability Index (HSI)** model:

```python
from VegProcessor.hsi import HSI

# Initialize the HSI model
hsi = HSI(config_file="./configs/hsi_config.yaml")

# Run the model
hsi.run()
hsi.post_process()  # optionally produce summaries
```

These steps are also demonstrated in `./VegProcessor/run.ipynb`.

---

### Option 2: Batch Execution with `batch_run.py`

For running multiple scenarios (e.g., multiple hydrologic models, SLR conditions, and flow scenarios), use `batch_run.py`. This script automates the execution and validation of multiple model runs.

#### Setting Up Batch Runs

1. **Edit the config file lists** in `VegProcessor/batch_run.py`:

```python
veg_config_files = [
    "/path/to/configs/veg_d3d_config_base_dry.yaml",
    "/path/to/configs/veg_d3d_config_base_wet.yaml",
    "/path/to/configs/veg_hec_config_1-08ft_slr_dry.yaml",
    # Add more config files as needed
]

hsi_config_files = [
    "/path/to/configs/hsi_d3d_config_base_dry.yaml",
    "/path/to/configs/hsi_d3d_config_base_wet.yaml",
    "/path/to/configs/hsi_hec_config_1-08ft_slr_dry.yaml",
    # Add more config files as needed
]
```

2. **Run the batch script**:

```bash
cd VegProcessor
python batch_run.py
```

3. **Follow the interactive prompts**:

```
Do you want to run Veg models? (y/n): y
Do you want to run HSI models? (y/n): y
```

If you answer "n" to both, you'll be prompted to validate existing outputs instead:

```
Do you want to validate existing outputs? (y/n): y
```

#### What Batch Run Does

The batch script performs the following for each config file:

1. **Executes the model** (`VegTransition` or `HSI`) with the specified config
2. **Calls `post_process()`** to generate summaries
3. **Validates outputs** after all runs complete:
   - Checks that NetCDF output files exist
   - Verifies the time dimension has the expected number of timesteps
   - Confirms the last timestep contains valid data (not all NaN)
4. **Prints a summary** showing successful and failed runs with error details

#### Example Output

```
Running VegTransition model for config: /path/to/veg_config_base_dry.yaml
Successfully completed VegTransition model for: /path/to/veg_config_base_dry.yaml

Validating VegTransition outputs...

============================================================
BATCH RUN RESULTS SUMMARY
============================================================

--- VegTransition Results ---

Successful (4/4):
  ✓ veg_config_base_dry.yaml
  ✓ veg_config_base_wet.yaml
  ✓ veg_config_1-08ft_slr_dry.yaml
  ✓ veg_config_1-08ft_slr_wet.yaml

--- HSI Results ---

Successful (3/4):
  ✓ hsi_config_base_dry.yaml
  ✓ hsi_config_base_wet.yaml
  ✓ hsi_config_1-08ft_slr_dry.yaml

Failed (1/4):
  ✗ hsi_config_1-08ft_slr_wet.yaml
    Reason: Time dim mismatch: expected 26, got 15
    Last log entries:
      2024-01-15 14:32:01 - Processing year 15...
      2024-01-15 14:32:45 - ERROR: Memory allocation failed

============================================================
```

#### Validation-Only Mode

To validate existing outputs without running any new models:

```bash
python batch_run.py
# Answer 'n' to both run prompts, then 'y' to validate
```

This is useful for:
- Checking the status of completed runs
- Debugging failed runs by examining log entries
- Verifying outputs after system interruptions

#### Execution Order

The batch script enforces the correct execution order:

1. All `VegTransition` models run first (if selected)
2. All `HSI` models run second (if selected)

This ensures HSI models have access to the required VegTransition outputs.

---

## Debugging & Logs

- Logs are stored in `output/run-metadata/simulation.log`
- Check logs if the model fails to run or if there are errors in output files.
- If running `VegTransition` or `HSI` in a notebook, the class instance (i.e. `hsi` as defined above) holds all of the intermediate and QA/QC arrays as attributes. For example: `hsi.alligator.si_1` is the location of suitability index #1 array for alligator. This array be be visualized by:

    ```python
    import matplotlib.pyplot as plt

    plt.matshow(hsi.alligator.si_1) # np.ndarray
    plt.colorbar()
    ```

---

## Cleaning Up

To deactivate the Conda environment:

```bash
conda deactivate
```

To remove the environment completely:

```bash
conda remove --name cpra_env --all -y
```
___
#### VegTransition
![alt text](./fig.png)