# CPRA HSI & Vegetation Modeling Tasks
___
This repo contains code for the Habitat Suitability Modeling (HSI) task, and the vegetation transition model, which serves as a partial basis for the HSI data input.

The processing code will be designed to facillitate these tasks:

1. General Data Preprocessing & Model Conceptualization
2. Calculate Transitions
3. Pre-process HSI Model inputs (e.g., Bald eagle)
4. Build/script HSI models 
5. Run HSI models output at each timestep/period
6. Visualize Output

___
### `VegProcessor/`
Description: This folder contains code for processing inputs and executing the vegetation transition model, as well as the subsequent HSI model implementation. It handles input data preprocessing, vegetation type transitions based on environmental conditions (e.g., water depth, salinity), and generating outputs for analysis and visualization. The `VegTransition` model is designed to simulate vegetation dynamics over time and provide inputs for Habitat Suitability Index (HSI) models.

#### Contents:
- `veg_transition.py`: Core logic for vegetation transition modeling, implementing rules and conditions for vegetation type changes over time. The `VegTransition` class is initialized with a `config.yaml` which defines the model parameters. The `run()` method executes the model.
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
#### `VegProcessor/sequences`
Description: this folder contains the individual species HSI logic.
#### Contents:
- `alligator.py`
- `bald_eagle.py`

___
#### VegTransition
![alt text](./fig.png)

