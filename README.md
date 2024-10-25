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
Description: This folder contains code for processing inputs to the vegetation transition model, as well as the model code itself.
#### Contents:
- `preprocess.py`
- `veg_transition.py`


### `HSI/`
Description: this folder contains code to setup and run individual HSI models.
#### Contents:
- `AlligatorHSI`: Alligator model.


![alt text](VegProcessor/hsi_model_diagram_export.png)

