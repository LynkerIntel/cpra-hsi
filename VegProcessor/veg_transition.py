import logging
import yaml
import xarray as xr
import numpy as np

import veg_logic


class VegTransition:
    """The Vegetation Transition Model."""

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

        Parameters:
        - config_file (str): Path to configuration YAML
        - log_level (int): Level of vebosity for logging.
        """
        # Set up the logger
        self._setup_logger(log_level)

        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # fetch raster data paths
        self.dem_path = config["raster_data"].get("dem_path")
        self.veg_base_path = config["raster_data"].get("veg_base_raster")

        # Extract initial state variables from config
        self.P = config["initial_conditions"].get("P0", 40)
        self.H = config["initial_conditions"].get("H0", 9)

        # Extract parameters from config
        self.alpha = config["parameters"].get("alpha", 0.1)
        self.beta = config["parameters"].get("beta", 0.02)

        # simulation parameters
        self.simulation_duration = config["simulation"].get("duration", 200)
        self.simulation_time_step = config["simulation"].get("time_step", 0.01)

        # Time
        self.time = 0

        # Store history for analysis
        self.history = {"P": [self.P], "H": [self.H], "time": [self.time]}

        # Log the initialization using lazy formatting
        self._logger.info(
            "Initialized model with P0=%s, H0=%s, alpha=%s, beta=%s, gamma=%s, delta=%s",
            self.P,
            self.H,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
        )

        # Load veg base and use as template to create arrays for the main state variables
        self.load_veg_base_raster()

        # create empty arrays for state variables, based on x, y dims of veg type base raster
        template = np.zeros((self.veg_type.ny, self.veg_type.nx))

        self.veg_type = template
        self.maturity = template
        self.dem = template
        self.pct_mast_hard = template
        self.pct_mast_soft = template
        self.pct_no_mast = template

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

    def step(self, dt=0.01):
        """Advance the transition model by one step."""
        # Calculate the rates of change (DUMMY VARS)
        dP = (self.alpha * self.P - self.beta * self.P * self.H) * dt
        dH = (self.delta * self.P * self.H - self.gamma * self.H) * dt

        # Update the state variables (DUMMY VARS)
        self.P += dP
        self.H += dH

        # example of how to call veg logic module
        self.veg_type = veg_logic.veg_logic(
            self.dem,
            self.pct_mast_hard,
            self.maturity,
        )

        # TODO: update for actual timestep, assuming year now
        self.maturity += 1

        # Update time
        self.time += dt

        # Record the new state
        self.history["P"].append(self.P)
        self.history["H"].append(self.H)
        self.history["time"].append(self.time)

        # Log the current state using lazy formatting
        self._logger.debug(
            "Time: %.2f, Prey Population: %.2f, Predator Population: %.2f",
            self.time,
            self.P,
            self.H,
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

    def load_dem(self):
        """ """
        # self.dem = load
        raise NotImplementedError

    def load_veg_base_raster(self):
        """This method will load the base veg raster, from which the model will iterate forwards,
        according to the transition logic.
        """
        ds = xr.open_dataset(self.veg_base_path)
        self.veg_type = ds["veg"]
