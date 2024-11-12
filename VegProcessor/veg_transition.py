import logging
import yaml
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import glob
import os
from typing import Optional
import rioxarray

import veg_logic
import hydro_logic


class VegTransition:
    """The Vegetation Transition Model.

    Vegetation zones are calculated independenlty (slower) for
    better model interperability and quality control.
    """

    def __init__(self, config_file, log_level=logging.INFO):
        """
        Initialize by setting up logger, loading resource paths, and creating empty
        arrays for state variables. State variables are:

        self.veg_type
        self.maturity
        self.elevation
        self.pct_mast_hard
        self.pct_mast_soft
        self.pct_no_mast
        self.salinity

        Parameters:
        - config_file (str): Path to configuration YAML
        - log_level (int): Level of vebosity for logging.
        """
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # fetch raster data paths
        self.dem_path = config["raster_data"].get("dem_path")
        self.wse_directory_path = config["raster_data"].get("wse_directory_path")
        self.veg_base_path = config["raster_data"].get("veg_base_raster")
        self.veg_keys_path = config["raster_data"].get("veg_keys")
        self.salinity_path = config["raster_data"].get("salinity_raster")

        # Extract initial state variables (dummy vars) from config
        self.alpha = config["parameters"].get("alpha", 0.1)
        self.beta = config["parameters"].get("beta", 0.02)

        # simulation parameters
        self.simulation_duration = config["simulation"].get("duration")
        self.simulation_time_step = config["simulation"].get("time_step")

        # Time
        self.time = 0

        # Store history for analysis
        # self.history = {"P": [self.P], "H": [self.H], "time": [self.time]}

        # Set up the logger
        self._setup_logger(log_level)

        # Load veg base and use as template to create arrays for the main state variables
        self.veg_type = self._load_veg_initial_raster()
        self.veg_keys = self._load_veg_keys()

        # self.zone_v = self.veg_type[self.veg_type == 23]

        # create empty arrays for state variables, based on x, y dims of veg type base raster
        # template = np.zeros((self.veg_type.ny, self.veg_type.nx))

        self.dem = self._load_dem()

        # load raster if provided, default values if not
        # self.load_salinity()

        self.wse = None
        self.maturity = np.ones_like(self.dem["band_data"].to_numpy())
        self.water_depth = None
        # self.pct_mast_hard = template
        # self.pct_mast_soft = template
        # self.pct_no_mast = template

    def _setup_logger(self, log_level):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("VegTransition")
        self._logger.setLevel(log_level)

        # Prevent adding multiple handlers if already added
        if not self._logger.handlers:
            # Create console handler and set level
            ch = logging.StreamHandler()
            ch.setLevel(log_level)

            # Create formatter and add it to the handler
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)

            # Add the handler to the logger
            self._logger.addHandler(ch)

    def step(self, date):
        """Advance the transition model by one step."""
        # get existing veg types
        veg_type_in = self.veg_type

        # calculate depth
        self.wse = self._load_wse_timestep(date=date, variable_name="WSE_MEAN")
        self.water_depth = self.get_depth()
        self._logger.info("Created depth for %s", date)

        # veg_type array is iteratively updated, for each zone
        self.veg_type = veg_logic.zone_v(self.veg_type, self.water_depth)
        # self.veg_type = veg_logic.zone_iv(self.veg_type, self.depth)

        # self.veg_type = veg_logic.zone_iv(self.veg_type, self.depth)
        # self.veg_type = veg_logic.zone_iv(self.veg_type, self.depth)
        # self.veg_type = veg_logic.zone_iv(self.veg_type, self.depth)
        # self.veg_type = veg_logic.zone_iv(self.veg_type, self.depth)
        # self.veg_type = veg_logic.zone_iv(self.veg_type, self.depth)
        # self.veg_type = veg_logic.zone_iv(self.veg_type, self.depth)
        # self.veg_type = veg_logic.zone_iv(self.veg_type, self.depth)

        # combine all zones into new timestep
        # must first ensure that there are no overlapping
        # values. Need a good QC method here.
        # self.veg_type = self.new_veg_1 + self.new_veg_2

        # if veg type has changed maturity = 0,
        # if veg type has not changes, maturity + 1
        self._calculate_maturity(veg_type_in)

        # Save all state variable arrays

        self._logger.debug(
            "Time: %.2f, var1: %.2f, var2: %.2f",
            self.time,
            self.dummy_var,
            self.dummy_var,
        )

    def run(self):
        """
        Run the vegetation transition model, with parameters defined in the configuration file.
        """
        # load data

        # create numpy base array

        # run model forwards
        steps = int(self.simulation_duration / self.simulation_time_step)
        self._logger.info(
            "Starting simulation for %s time units with time step %s",
            self.simulation_duration,
            self.simulation_time_step,
        )

        for _ in range(steps):
            self.step(self.simulation_time_step)

        self._logger.info("Simulation complete")

    def _load_dem(self) -> xr.Dataset:
        """Load project domain DEM."""
        dem = xr.open_dataset(self.dem_path)
        self._logger.info("Loaded DEM")
        return dem

    def _load_wse_timestep(
        self,
        date: str,
        variable_name: str = "WSE_MEAN",
        date_format: str = "%Y_%m_%d",
    ) -> Optional[xr.DataArray]:
        """
        Load a single timestep from a folder of .tif files as an xarray.DataArray.

        The .tif file should have a filename that includes a variable name (e.g., `WSE_MEAN`)
        followed by a timestamp in the specified format (e.g., `2005_10_01`). The function will
        locate the file corresponding to the specified date and load it as a DataArray.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the .tif files.
        date : str
            The date for the timestep to load, formatted according to `date_format`.
        variable_name : str, optional
            The name of the variable to use in the DataArray.
        date_format : str, optional
            Format string for parsing dates from file names, default is "%Y_%m_%d".
            Adjust based on your file naming convention.

        Returns
        -------
        xr.DataArray or None
            An xarray.DataArray with the raster data for the specified timestep,
            or None if the file is not found.
        """
        # Format the date into the specified date_format to match the file names
        date_str = pd.to_datetime(date).strftime(date_format)

        # Construct the expected filename pattern
        expected_filename = f"{variable_name}_{date_str}.tif"
        file_path = os.path.join(self.wse_directory_path, expected_filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            self._logger.info("File not found: %s", file_path)
            return None

        # Load the .tif file as a DataArray
        da = rioxarray.open_rasterio(file_path).squeeze(dim="band")

        # Rename the variable and add the time coordinate
        da = da.rename(variable_name).expand_dims(time=[pd.to_datetime(date)])

        return da

    def _get_depth(self):
        """Calculate water depth from DEM and Water Surface Elevation."""
        return self.wse - self.dem

    def _calculate_maturity(self, veg_type_in: np.ndarray):
        """
        +1 maturity for pixels without vegetation changes.
        """
        # TODO: Need to create mask for only Zone II to V pixels.
        self._logger.info("WARNING, need to mask to zone II to V pixels.")

        # get inverse of "equals" element comparison,
        # i.e. True where elements are different
        mask = ~np.equal(veg_type_in, self.veg_type)
        self.maturity[mask] += 1
        self._logger.info("Maturity incremented for unchanged veg types")

    def load_landcover(self) -> np.ndarray:
        """This method will load the landcover dataset, which may
        be needed?
        """
        raise NotImplementedError

    def _load_veg_initial_raster(self):
        """This method will load the base veg raster, from which the model will iterate forwards,
        according to the transition logic.

        """
        da = xr.open_dataarray(self.veg_base_path)
        return da

    def _load_veg_keys(self) -> pd.DataFrame:
        """load vegetation class names from database file"""
        dbf = gpd.read_file(self.veg_keys_path)
        # fix dtype
        dbf["Value"] = dbf["Value"].astype(int)
        return dbf

    def _load_salinity(self):
        """Load salinity raster data (if available.)"""
        # raise NotImplementedError
        if self.salinity_path:
            self.salinity = None
            self._logger.info("Loaded salinity from raster")
        else:
            self.salinity = hydro_logic.habitat_based_salinity(self.veg_type)

            self._logger.info("Creating salinity from habitat defaults")
