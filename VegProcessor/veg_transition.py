import logging
import yaml
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd

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

        # # Log the initialization using lazy formatting
        # self._logger.info(
        #     "Initialized model with P0=%s, H0=%s, alpha=%s, beta=%s,
        #     self.dummy_var,
        #     self.dummy_var,
        #     self.dummy_var,
        #     self.dummy_var,
        # )

        # Load veg base and use as template to create arrays for the main state variables
        self.veg_type = self.load_veg_initial_raster()
        self.veg_keys = self.load_veg_keys()

        # self.zone_v = self.veg_type[self.veg_type == 23]

        # create empty arrays for state variables, based on x, y dims of veg type base raster
        # template = np.zeros((self.veg_type.ny, self.veg_type.nx))

        self.dem  # = load dem
        self.wse  # = load wse
        # caluculate depth
        self.depth = self.get_depth(self.dem, self.wse)

        # load raster if provided, default values if not
        # self.load_salinity()

        # self.veg_type = template
        # self.maturity = template
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

    def step(self, dt=1):
        """Advance the transition model by one step."""
        # calculate inundation

        # example of how to call veg logic module
        self.veg_type = veg_logic.zone_v(self.veg_type, self.depth)
        self.veg_type = veg_logic.zone_iv(self.veg_type, self.depth)

        self.new_veg_3

        # combine all zones into new timestep
        # must first ensure that there are no overlapping
        # values. Need a good QC method here.
        self.veg_type = self.new_veg_1 + self.new_veg_2

        self.create_new_veg_arrays()

        # if veg type has changed maturity = 0,
        # if veg type has not changes, maturity + 1

        # TODO: update for actual timestep, assuming year now
        self.maturity += dt

        # Update time
        self.time += dt

        # Record the new state
        self.history["P"].append(self.P)
        self.history["H"].append(self.H)
        self.history["time"].append(self.time)

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

    def load_dem(self) -> np.ndarray:
        """Load project domain DEM."""
        # self.dem = load
        raise NotImplementedError

    def load_landcover(self) -> np.ndarray:
        """This method will load the landcover dataset, which may
        be needed?
        """
        raise NotImplementedError

    def load_veg_initial_raster(self):
        """This method will load the base veg raster, from which the model will iterate forwards,
        according to the transition logic.

        """
        da = xr.open_dataarray(self.veg_base_path)
        return da

    def load_veg_keys(self) -> pd.DataFrame:
        """load vegetation class names from database file"""
        dbf = gpd.read_file(self.veg_keys_path)
        # fix dtype
        dbf["Value"] = dbf["Value"].astype(int)
        return dbf

    def load_salinity(self):
        """Load salinity raster data (if available.)"""
        # raise NotImplementedError
        if self.salinity_path:
            self.salinity = None
            self._logger.info("Loaded salinity from raster")
        else:
            self.salinity = hydro_logic.habitat_based_salinity(self.veg_type)

            self._logger.info("Creating salinity from habitat defaults")
