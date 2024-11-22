import logging
import yaml
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import shutil
import glob
import os
from typing import Optional
import rioxarray
from datetime import datetime

import veg_logic
import hydro_logic
import plotting
import testing
import utils


class VegTransition:
    """The Vegetation Transition Model.

    Vegetation zones are calculated independenlty (slower) for
    better model interperability and quality control.

    Notes: two dimensinal np.ndarray are used as default, with xr.Dataset for cases
    where time series arrays are needed. vegetation numpy arrays are dtype float32,
    because int types have limited interoperability with the np.nan type, which is needed.
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
        self.sim_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config_path = config_file

        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

        # fetch raster data paths
        self.dem_path = self.config["raster_data"].get("dem_path")
        self.wse_directory_path = self.config["raster_data"].get("wse_directory_path")
        self.veg_base_path = self.config["raster_data"].get("veg_base_raster")
        self.veg_keys_path = self.config["raster_data"].get("veg_keys")
        self.salinity_path = self.config["raster_data"].get("salinity_raster")

        # simulation
        self.start_date = self.config["simulation"].get("start_date")
        self.end_date = self.config["simulation"].get("end_date")

        # output
        self.output_base_dir = self.config["output"].get("output_base")

        # Pretty-print the configuration
        config_pretty = yaml.dump(
            self.config, default_flow_style=False, sort_keys=False
        )

        # Time
        # self.time = 0

        # Store history for analysis
        # self.history = {"P": [self.P], "H": [self.H], "time": [self.time]}

        # Set up the logger
        self._setup_logger(log_level)

        # Pretty-print the configuration
        config_pretty = yaml.dump(
            self.config, default_flow_style=False, sort_keys=False
        )

        # Log the configuration
        self._logger.info("Loaded Configuration:\n%s", config_pretty)

        # setup output dir
        self.create_output_dirs()

        self.dem = self._load_dem()
        # print(self.dem.shape)
        # Load veg base and use as template to create arrays for the main state variables
        self.veg_type = self._load_veg_initial_raster()
        self.veg_keys = self._load_veg_keys()

        # load raster if provided, default values if not
        self._load_salinity()

        self.wse = None
        self.maturity = np.ones_like(self.dem)  # TODO: should maturity iterate from 0?
        self.water_depth = None
        self.veg_ts_out = None  # xarray output for timestep

        # initialize partial update arrays as None
        self.veg_type_update_1 = None
        self.veg_type_update_2 = None
        self.veg_type_update_3 = None
        self.veg_type_update_4 = None
        self.veg_type_update_5 = None
        self.veg_type_update_6 = None
        self.veg_type_update_7 = None
        self.veg_type_update_8 = None
        self.veg_type_update_9 = None
        self.veg_type_update_10 = None

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
        self._logger.info("starting timestep: %s", date)
        self._create_timestep_dir(date)

        wy = date.year
        # copy existing veg types
        veg_type_in = self.veg_type

        # calculate depth
        self.wse = self._load_wse_wy(wy, variable_name="WSE_MEAN")
        self.wse = self._reproject_match_to_dem(self.wse)  # TEMPFIX
        self.water_depth = self._get_depth()

        # temporary bug fixing subset
        # self.veg_type = self.veg_type[200:275, 200:275]
        # self.water_depth = self.water_depth.isel(x=slice(200, 275), y=slice(200, 275))

        plotting.np_arr(
            self.veg_type,
            title="All Types Input",
            out_path=self.timestep_output_dir,
        )

        # veg_type array is iteratively updated, for each zone
        self.veg_type_update_1 = veg_logic.zone_v(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            date,
            # plot=True,
        )
        self.veg_type_update_2 = veg_logic.zone_iv(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            date,
            # plot=True,
        )
        self.veg_type_update_3 = veg_logic.zone_iii(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            date,
            # plot=True,
        )
        self.veg_type_update_4 = veg_logic.zone_ii(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            date,
            # plot=True,
        )
        self.veg_type_update_5 = veg_logic.fresh_shrub(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            date,
            # plot=True,
        )
        self.veg_type_update_6 = veg_logic.fresh_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            self.salinity,
            date,
            # plot=True,
        )
        self.veg_type_update_7 = veg_logic.intermediate_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            self.salinity,
            date,
            # plot=True,
        )
        self.veg_type_update_8 = veg_logic.brackish_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            self.salinity,
            date,
            # plot=True,
        )
        self.veg_type_update_9 = veg_logic.saline_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            self.salinity,
            date,
            # plot=True,
        )
        self.veg_type_update_10 = veg_logic.water(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir,
            self.salinity,
            date,
            # plot=True,
        )

        # stack partial update arrays for each zone
        stacked_veg = np.stack(
            (
                self.veg_type_update_1,
                self.veg_type_update_2,
                self.veg_type_update_3,
                self.veg_type_update_4,
                self.veg_type_update_5,
                self.veg_type_update_6,
                self.veg_type_update_7,
                self.veg_type_update_8,
                self.veg_type_update_9,
                self.veg_type_update_10,
            )
        )

        if testing.has_overlapping_non_nan(stacked_veg):
            raise ValueError(
                "New vegetation stacked arrays cannot have overlapping values."
            )

        # combine arrays while preserving NaN
        self._logger.info("Combining new vegetation types into full array.")
        self.veg_type = np.full_like(self.veg_type_1, np.nan)  # Initialize with NaN
        for layer in stacked_veg:
            self.veg_type = np.where(np.isnan(self.veg_type), layer, self.veg_type)

        # get unchanged/unhandled vegetation types from base raster
        no_transition_nan_mask = np.isnan(self.veg_type)

        # Replace NaN values in the new array with corresponding values from the veg base raster
        self._logger.info(
            "Filling NaN pixels in result array with vegetation base raster."
        )
        self.veg_type[no_transition_nan_mask] = self._load_veg_initial_raster()[
            no_transition_nan_mask
        ]

        plotting.np_arr(
            self.veg_type,
            title="All Types Output",
            out_path=self.timestep_output_dir,
        )
        plotting.sum_changes(
            veg_type_in,
            self.veg_type,
            plot_title="Timestep Veg Changes",
            out_path=self.timestep_output_dir,
        )

        # if veg type has changed maturity = 0,
        # if veg type has not changes, maturity + 1
        # CURRENTLY HAS BUG
        self._calculate_maturity(veg_type_in)

        # serialize state variables: veg_type, maturity, mast %
        self._logger.info("saving state variables for timestep.")
        self._save_state_vars(date)

        self._logger.info("completed timestep: %s", date)

    def run(self):
        """
        Run the vegetation transition model, with parameters defined in the configuration file.
        """
        # run model forwards
        # steps = int(self.simulation_duration / self.simulation_time_step)
        self._logger.info(
            "Starting simulation at %s. Period: %s - %s",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.start_date,
            self.end_date,
        )
        simulation_period = pd.date_range(self.start_date, self.end_date, freq="y")
        self._logger.info("Running model for: %s timesteps", len(simulation_period))

        for timestep in simulation_period:
            self.step(timestep)

        self._logger.info("Simulation complete")

    def _load_dem(self) -> np.ndarray:
        """Load project domain DEM."""
        ds = xr.open_dataset(self.dem_path)
        ds = ds.squeeze(drop="band_data")
        da = ds.to_dataarray(dim="band")
        self._logger.info("Loaded DEM")
        # TODO: where is extra dim coming from? i.e. da[0] is needed!
        return da[0].to_numpy()

    def _load_wse_wy(
        self,
        water_year: int,
        variable_name: str = "WSE_MEAN",
        date_format: str = "%Y_%m_%d",
    ) -> Optional[xr.Dataset]:
        """
        Load .tif files corresponding to a specific water year into an xarray.Dataset.

        Each .tif file should have a filename that includes a variable name (e.g., `WSE_MEAN`)
        followed by a timestamp in the specified format (e.g., `2005_10_01`). The function will
        automatically extract the timestamps and assign them to a 'time' dimension in the resulting
        xarray.Dataset, but only for files within the specified water year.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the .tif files.
        water_year : int
            The water year to filter files by. Files from October 1 of the previous year
            to September 30 of this year will be loaded.
        variable_name : str, optional
            The name of the variable to use in the dataset.
        date_format : str, optional
            Format string for parsing dates from file names, default is "%Y_%m_%d".
            Adjust based on your file naming convention.

        Returns
        -------
        xr.Dataset or None
            An xarray.Dataset with the raster data from each .tif file within the water year,
            stacked along a 'time' dimension, with the specified variable name.
            Returns None if no files are found for the specified water year.
        """
        tif_files = sorted(
            glob.glob(os.path.join(self.wse_directory_path, "**/*.tif"), recursive=True)
        )

        start_date = pd.to_datetime(f"{water_year - 1}-10-01")
        end_date = pd.to_datetime(f"{water_year}-09-30")

        selected_files = []
        time_stamps = []

        for f in tif_files:
            date_str = "_".join(os.path.basename(f).split("_")[2:5]).replace(".tif", "")
            file_date = pd.to_datetime(date_str, format=date_format)

            if start_date <= file_date <= end_date:
                selected_files.append(f)
                time_stamps.append(file_date)

        if not selected_files:
            self._logger.error("No files found for water year: %s", water_year)
            return None

        # Preprocess function to remove the 'band' dimension
        def preprocess(da):
            return da.squeeze(dim="band").expand_dims(
                time=[time_stamps[selected_files.index(da.encoding["source"])]]
            )

        # Load selected files into a single Dataset with open_mfdataset
        xr_dataset = xr.open_mfdataset(
            selected_files,
            concat_dim="time",
            combine="nested",
            parallel=True,
            preprocess=preprocess,
        )

        # rename
        xr_dataset = xr_dataset.rename(
            {list(xr_dataset.data_vars.keys())[0]: variable_name}
        )

        self._logger.info("Loaded HEC-RAS WSE Datset for year: %s", water_year)
        return xr_dataset

    def _reproject_match_to_dem(
        self, ds: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        """
        Temporary fix to match WSE model output to 60m DEM grid.
        """
        ds_dem = xr.open_dataset(self.dem_path)
        ds_dem = ds_dem.squeeze(drop="band_data")
        da_dem = ds_dem.to_dataarray(dim="band")

        # self._logger.warning("reprojecting %s to match DEM. TEMPFIX!", ds.variables)
        return ds.rio.reproject_match(da_dem)

    def _get_depth(self) -> xr.Dataset:
        """Calculate water depth from DEM and Water Surface Elevation.

        TODO: check units !
        """
        self._logger.info("Creating depth")
        return self.wse - self.dem

    def _calculate_maturity(self, veg_type_in: np.ndarray):
        """
        +1 maturity for pixels without vegetation changes.
        """
        # TODO: Need to create mask for only Zone II to V pixels.
        # self._logger.warning("need to mask to zone II to V pixels.")

        # Ensure both arrays have the same shape
        if veg_type_in.shape != self.veg_type.shape:
            raise ValueError("Timestep input and output array have different shapes!")

        # create a boolean array where True indicates elements are different
        # Use np.isnan to handle NaN values specifically (they don't indicate
        # a veg transition has occurred)
        diff_mask = (veg_type_in != self.veg_type) & ~(
            np.isnan(veg_type_in) & np.isnan(self.veg_type)
        )

        self.maturity[diff_mask] += 1
        self._logger.info("Maturity incremented for unchanged veg types")

    def load_landcover(self) -> np.ndarray:
        """This method will load the landcover dataset, which may
        be needed?
        """
        raise NotImplementedError

    def _load_veg_initial_raster(self) -> np.ndarray:
        """This method will load the base veg raster, from which the model will iterate forwards,
        according to the transition logic.
        """
        da = xr.open_dataarray(self.veg_base_path)
        da = da.squeeze(drop="band")

        da = self._reproject_match_to_dem(da)
        self._logger.info("Loaded initial vegetation raster")
        return da.to_numpy()

    def _load_veg_keys(self) -> pd.DataFrame:
        """load vegetation class names from database file"""
        dbf = gpd.read_file(self.veg_keys_path)
        # fix dtype
        dbf["Value"] = dbf["Value"].astype(int)
        self._logger.info("Loaded Vegetation Keys")
        return dbf

    def _load_salinity(self):
        """Load salinity raster data (if available.)"""
        # raise NotImplementedError
        if self.salinity_path:
            # add loading code here
            self._logger.info("Loaded salinity from raster")
        else:
            self.salinity = hydro_logic.habitat_based_salinity(self.veg_type)
            self._logger.info("Creating salinity from habitat defaults")

    def create_output_dirs(self):
        """Create an output location for state variables, model config,
        input data, and QC plots.
        """
        output_dir_name = f"VegOut_{self.sim_start_time}"

        # Combine base directory and new directory name
        self.output_dir_path = os.path.join(self.output_base_dir, output_dir_name)
        # Create the directory if it does not exist
        os.makedirs(self.output_dir_path, exist_ok=True)
        self._logger.info("Created output directory at %s", self.output_dir_path)

        # Create the 'run-input' subdirectory
        run_input_dir = os.path.join(self.output_dir_path, "run-input")
        os.makedirs(run_input_dir, exist_ok=True)
        self._logger.info("Created 'run-input' directory at %s", run_input_dir)

        # Copy the config YAML file to 'run-input'
        # config_file_path = (
        #     self.config_file_path
        # )  # Assuming config_file_path is an attribute
        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, run_input_dir)
            self._logger.info(
                "Copied config YAML file from %s to %s", self.config_path, run_input_dir
            )
        else:
            self._logger.warning("Config file not found at %s", self.config_path)

    def _save_state_vars(self, date):
        """The method will save state variables after each timestep.

        This method should also include the config, input data, and QC plots.

        """
        template = self.water_depth.isel({"time": 0})  # subset to first month

        # veg type out
        new_variables = {"veg_type": (self.veg_type, {"units": "veg_type"})}
        self.veg_ts_out = utils.create_dataset_from_template(template, new_variables)

        outpath = self.timestep_output_dir + "/vegtype.tif"
        self.veg_ts_out["veg_type"].rio.to_raster(outpath)

        # pct mast out

        # maturity out

    def _create_timestep_dir(self, date):
        """ """
        self.timestep_output_dir = os.path.join(
            self.output_dir_path, f"{date.strftime('%Y%m%d')}"
        )
        os.makedirs(self.timestep_output_dir, exist_ok=True)
        os.makedirs(self.timestep_output_dir + "/figs", exist_ok=True)
