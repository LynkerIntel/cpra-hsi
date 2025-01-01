import logging
import yaml
import subprocess
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import shutil
import glob
import os
from typing import Optional
import rioxarray  # used for tif output
from datetime import datetime
import matplotlib.pyplot as plt


import veg_logic
import hydro_logic
import plotting
import utils


class VegTransition:
    """The Vegetation Transition Model.

    Vegetation zones are calculated independently, then combined into single
    non-overlapping array for each timestep.

    Example usage found in `./run.ipynb`

    Notes: two dimensinal np.ndarray are used as default, with xr.Dataset for cases
    where time series awareness is helpful (i.e. subsetting WSE data). vegetation numpy
    arrays are dtype float32, because int types have limited interoperability with the
    np.nan type, which is needed.


    Attributes: (only state variables listed)
    -----------
        self.veg_type
        self.maturity
        self.elevation
        self.pct_mast_hard
        self.pct_mast_soft
        self.pct_no_mast
        self.salinity

    Methods
    -------
    step(date):
        Advances the vegetation model by a single timestep.
    run():
        Run the vegetation model, using settings and data defined in `veg_config.yaml`

    """

    def __init__(self, config_file: str, log_level: int = logging.INFO):
        """
        Initialize by setting up logger, loading resource paths, and creating empty
        arrays for state variables. State variables are:

        Parameters:
        -----------
        config_file : str
            Path to configuration YAML
        log_level : int
            Level of vebosity for logging.
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
        self.water_year_start = self.config["simulation"].get("water_year_start")
        self.water_year_end = self.config["simulation"].get("water_year_end")
        self.analog_sequence = self.config["simulation"].get("wse_sequence_input")

        # metadata
        self.metadata = self.config["metadata"]
        self.scenario_type = self.config["metadata"].get(
            "scenario", ""
        )  # empty str if missing

        # self.model = self.config["metadata"].get("model")
        # self.group = self.config["metadata"].get("group")
        # self.wpu = self.config["metadata"].get("wpu")
        # self.ion = self.config["metadata"].get("ion")

        # output
        self.output_base_dir = self.config["output"].get("output_base")

        # Pretty-print the configuration
        config_pretty = yaml.dump(
            self.config, default_flow_style=False, sort_keys=False
        )

        self._create_output_dirs()
        self.current_timestep = None
        self._setup_logger(log_level)
        self.timestep_output_dir = None

        # Pretty-print the configuration
        config_pretty = yaml.dump(
            self.config, default_flow_style=False, sort_keys=False
        )

        # Log the configuration
        self._logger.info("Loaded Configuration:\n%s", config_pretty)
        self._get_git_commit_hash()

        self.dem = self._load_dem()
        # print(self.dem.shape)
        # Load veg base and use as template to create arrays for the main state variables
        self.veg_type = self._load_veg_initial_raster()
        self.veg_keys = self._load_veg_keys()

        self.wse = None
        self.maturity = np.zeros_like(self.dem)
        self.water_depth = None
        self.veg_ts_out = None  # xarray output for timestep
        self.salinity = None

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

    def _setup_logger(self, log_level=logging.INFO):
        """Set up the logger for the VegTransition class."""
        self._logger = logging.getLogger("VegTransition")
        self._logger.setLevel(log_level)

        # clear existing handlers to prevent duplicates
        # (this happens when re-runnning in notebook)
        if self._logger.hasHandlers():
            for handler in self._logger.handlers:
                self._logger.removeHandler(handler)
                handler.close()  # Close old handlers properly

        try:
            # console handler for stdout
            # i.e. print messages
            ch = logging.StreamHandler()
            ch.setLevel(log_level)

            # file handler for logs in `run-input` folder
            run_metadata_dir = os.path.join(self.output_dir_path, "run-metadata")
            os.makedirs(run_metadata_dir, exist_ok=True)  # Ensure directory exists
            log_file_path = os.path.join(run_metadata_dir, "simulation.log")
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(log_level)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [Timestep: %(timestep)s] - %(message)s"
            )

            # Add formatter to handlers
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)

            # Add handlers to the logger
            self._logger.addHandler(ch)
            self._logger.addHandler(fh)

            # Add a custom filter to inject the timestep
            filter_instance = _TimestepFilter(self)
            self._logger.addFilter(filter_instance)

            self._logger.info("Logger setup complete.")
        except Exception as e:
            print(f"Error during logger setup: {e}")

    def _get_git_commit_hash(self):
        """Retrieve the current Git commit hash for the repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(
                    self.config_path
                ),  # Ensure it's run in the repo directory
                capture_output=True,
                text=True,
                check=True,
            )
            result = result.stdout.strip()
            self._logger.info(f"Model code version from git: {result}")

        except subprocess.CalledProcessError as e:
            self._logger.warning("Unable to fetch Git commit hash: %s", e)
            return "unknown"

    def step(self, timestep: pd.DatetimeTZDtype, counter: int, simulation_period: int):
        """Advance the transition model by one step.

        Parameters:
        -----------
            timestep : pd.DatetimeTZDtype
                The current model timestep.
            counter : int
                Integer representation of timestep.
            simulation_period: int
                The length of the simulation. i.e. 25 years for a single
                scenario run.
        """
        self.current_timestep = timestep  # Set the current timestep
        wy = timestep.year

        self._logger.info("starting timestep: %s", timestep)
        self._create_timestep_dir(counter.zfill(2))

        # copy existing veg types
        veg_type_in = self.veg_type

        # calculate depth
        self.wse = self.load_wse_wy(wy, variable_name="WSE_MEAN")
        self.wse = self._reproject_match_to_dem(self.wse)  # TEMPFIX
        self.water_depth = self._get_depth()

        # get salinity
        self.salinity = self._get_salinity()

        plotting.np_arr(
            self.veg_type,
            f"All Types Input {self.current_timestep.strftime('%Y-%m-%d')} {self.scenario_type}",
            out_path=self.timestep_output_dir_figs,
            veg_palette=True,
        )

        # veg_type array is iteratively updated, for each zone
        self.veg_type_update_1 = veg_logic.zone_v(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
        )
        self.veg_type_update_2 = veg_logic.zone_iv(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
        )
        self.veg_type_update_3 = veg_logic.zone_iii(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
        )
        self.veg_type_update_4 = veg_logic.zone_ii(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
        )
        self.veg_type_update_5 = veg_logic.fresh_shrub(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
        )
        self.veg_type_update_6 = veg_logic.fresh_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
        )
        self.veg_type_update_7 = veg_logic.intermediate_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
        )
        self.veg_type_update_8 = veg_logic.brackish_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
        )
        self.veg_type_update_9 = veg_logic.saline_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
        )
        self.veg_type_update_10 = veg_logic.water(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
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

        if utils.has_overlapping_non_nan(stacked_veg):
            raise ValueError(
                "New vegetation stacked arrays cannot have overlapping values."
            )

        # combine arrays while preserving NaN
        self._logger.info("Combining new vegetation types into full array.")
        self.veg_type = np.full_like(self.veg_type_update_1, np.nan)
        for layer in stacked_veg:
            self.veg_type = np.where(np.isnan(self.veg_type), layer, self.veg_type)

        # get unchanged/unhandled vegetation types from base raster
        # no_transition_nan_mask = np.isnan(self.veg_type)

        # Replace NaN values in the new array with corresponding values from the veg base raster
        # self._logger.info(
        #     "Filling NaN pixels in result array with vegetation base raster."
        # )
        # self.veg_type[no_transition_nan_mask] = self._load_veg_initial_raster()[
        #     no_transition_nan_mask
        # ]

        plotting.np_arr(
            self.veg_type,
            title=f"All Types Output {self.current_timestep.strftime('%Y-%m-%d')} {self.scenario_type}",
            out_path=self.timestep_output_dir_figs,
            veg_palette=True,
        )
        plotting.sum_changes(
            veg_type_in,
            self.veg_type,
            plot_title=f"Timestep Veg Changes {self.current_timestep.strftime('%Y-%m-%d')} {self.scenario_type}",
            out_path=self.timestep_output_dir_figs,
        )

        self._calculate_maturity(veg_type_in)

        # serialize state variables: veg_type, maturity, mast %
        self._logger.info("saving state variables for timestep.")

        params = {
            "model": self.metadata.get("model"),
            "scenario": self.metadata.get("scenario"),
            "group": self.metadata.get("group"),  # not sure if this is correct,
            "wpu": "ARS",
            "io_type": "O",
            "time_freq": "ANN",  # for annual output
            "year_range": f"{counter.zfill(2)}_{simulation_period.zfill(2)}",
            "parameter": "NA",  # ?
        }

        self._save_state_vars(params)

        self._logger.info("completed timestep: %s", timestep)
        self.current_timestep = None

    def run(self):
        """
        Run the vegetation transition model, with parameters defined in the configuration file.

        Start and end parameters are year, and handled as ints. No other frequency currently possible.
        """
        # run model forwards
        # steps = int(self.simulation_duration / self.simulation_time_step)
        self._logger.info(
            "Starting simulation at %s. Period: %s - %s",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.water_year_start,
            self.water_year_end,
        )

        # plus 1 to make inclusive
        simulation_period = range(self.water_year_start, self.water_year_end + 1)

        self._logger.info("Running model for: %s timesteps", len(simulation_period))

        for i, wy in enumerate(simulation_period):
            self.step(
                timestep=pd.to_datetime(f"{wy}-10-01"),
                counter=str(i + 1),
                simulation_period=str(len(simulation_period)),
            )

        self._logger.info("Simulation complete")
        logging.shutdown()

    def _load_dem(self) -> np.ndarray:
        """Load project domain DEM."""
        ds = xr.open_dataset(self.dem_path)
        ds = ds.squeeze(drop="band_data")
        da = ds.to_dataarray(dim="band")
        self._logger.info("Loaded DEM")
        # TODO: where is extra dim coming from? i.e. da[0] is needed!
        return da[0].to_numpy()

    def load_wse_wy(
        self,
        water_year: int,
        variable_name: str = "WSE_MEAN",
        date_format: str = "%Y_%m_%d",
    ) -> xr.Dataset:
        """
        Load .tif files corresponding to a specific water year into an xarray.Dataset. This method
        uses lazy-loading via Dask.

        Each .tif file should have a filename that includes a variable name (e.g., `WSE_MEAN`)
        followed by a timestamp in the specified format (e.g., `2005_10_01`). The function will
        automatically extract the timestamps and assign them to a 'time' dimension in the resulting
        xarray.Dataset, but only for files within the specified water year.

        NOTE: Input WSE data (as of 2024-12-13) uses NaN to designate 0 depth. If this is false,
        because the input data has changed, or a different model with different NaAn classification
        is used, this function must be updated accordingly.

        UNIT: Input raster is assumed to be feet, and returned in meters to match the DEM.

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
        analog_sequence : bool
            This is only for use when loading 25-year analog years sequences, where the
            filenames have been updated to represent analog years, but the file data
            reamain unchanged (i.e. uses the actual model-year). This flag will reset
            the MONTHLY time series to match the year given in the file name, in order
            for the vegetation model to run as normal.

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
            self._logger.error("No WSE files found for water year: %s", water_year)
            return None

        if len(selected_files) < 12:
            raise ValueError(f"month(s) missing from Water Year: {water_year}")

        # Preprocess function to remove the 'band' dimension
        def preprocess(da):
            return da.squeeze(dim="band").expand_dims(
                time=[time_stamps[selected_files.index(da.encoding["source"])]]
            )

        # Load selected files into a single Dataset with open_mfdataset
        ds = xr.open_mfdataset(
            selected_files,
            concat_dim="time",
            combine="nested",
            parallel=True,
            preprocess=preprocess,
        )
        # rename
        ds = ds.rename({list(ds.data_vars.keys())[0]: variable_name})

        ds[variable_name] *= 0.3048  # UNIT: feet to meters

        if self.analog_sequence:
            self._logger.info("Using sequence loading method.")
            new_timesteps = pd.date_range(
                f"{water_year-1}-10-01", f"{water_year}-09-01", freq="MS"
            )

            if not len(new_timesteps) == len(ds["time"]):
                raise ValueError("Timestep must be monthly.")

            # Replace the time coordinate with time from
            ds = ds.assign_coords(time=("time", new_timesteps))
            self._logger.info("Sequence timeseres updated to match filename.")

        self._logger.info("Loaded HEC-RAS WSE Datset for water-year: %s", water_year)
        return ds

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

    def _get_depth(self, annual_mean: bool = False) -> xr.Dataset:
        """Calculate water depth from DEM and Water Surface Elevation.

        NOTE: NaN values are changed to 0 after differencing, so that Null WSE becomes
        equivalent to 0 water depth. This is necessary so that inundation checks do
        not have NaN values for periods without inundation (logic relies
        on > or < comparison operators).
        """
        self._logger.info("Creating depth")
        ds = self.wse - self.dem

        self._logger.info(
            "Replacing all NaN in depth array with 0 (assuming full domain coverage.)"
        )
        # fill zeros. This is necessary to get 0 water depth from DEM and WSE!
        ds = ds.fillna(0)

        if annual_mean:
            ds = ds.mean(dim="time")

        # ds["WSE_MEAN"].plot(
        #     col="time",  # Create panels for each time step
        #     col_wrap=4,  # Number of panels per row
        #     cmap="viridis",
        #     aspect=1.5,  # Adjust aspect ratio
        #     size=3,  # Adjust figure size
        # )
        # plt.show()
        return ds

    def _calculate_maturity(self, veg_type_in: np.ndarray):
        """
        +1 year maturity for pixels without vegetation changes.
        """
        # Ensure both arrays have the same shape
        if veg_type_in.shape != self.veg_type.shape:
            raise ValueError("Timestep input and output array have different shapes!")

        # create a boolean array where True indicates elements are different
        # Use np.isnan to handle NaN values specifically (they don't indicate
        # a veg transition has occurred)
        diff_mask = (veg_type_in != self.veg_type) & ~(
            np.isnan(veg_type_in) & np.isnan(self.veg_type)
        )

        # forested veg types
        values_to_mask = [15, 16, 17, 18]
        # Create mask where True corresponds to values in the list
        type_mask = np.isin(self.veg_type, values_to_mask)

        # Use logical AND to find locations where both arrays are True
        # i.e. pixel value has changed, and it is a forested veg type
        stacked_mask_change = np.stack((diff_mask, type_mask))
        combined_mask_change = np.logical_and.reduce(stacked_mask_change)

        # pixel value has NOT changed, and it is a forested veg type
        stacked_mask_no_change = np.stack((~diff_mask, type_mask))
        combined_mask_no_change = np.logical_and.reduce(stacked_mask_no_change)

        if not combined_mask_no_change.any():
            self._logger.warning("No forested pixels had maturity increment increase.")

        if not combined_mask_change.any():
            self._logger.warning("No forested pixels changed in prior timestep.")

        if utils.common_true_locations(
            np.stack((combined_mask_change, combined_mask_no_change))
        ):
            raise ValueError("Forested types have overlapping True location(s)")

        # if forested pixels change, reset age to 0
        self.maturity[combined_mask_change] = 0
        self._logger.info("Maturity reset for changed veg type (forested)")
        # if forested pixels are the same, add one year
        self.maturity[combined_mask_no_change] += 1
        self._logger.info("Maturity incremented for unchanged veg types (forested)")

        # all other types (non-forested, non-handled) to np.nan
        self.maturity[~type_mask] = np.nan

        plotting.np_arr(
            self.maturity,
            title=f"Timestep Maturity {self.current_timestep.strftime('%Y-%m-%d')} {self.scenario_type}",
            out_path=self.timestep_output_dir_figs,
        )

    def _load_veg_initial_raster(self, xarray=False) -> np.ndarray | xr.Dataset:
        """This method will load the base veg raster, from which the model will iterate forwards,
        according to the transition logic.

        Parameters
        ----------
        xarray : bool
            True if xarray output format is needed.

        Returns
        -------
        xr.Dataarray or np.ndarray
            An array with the intial vegetation types, subset to types with transition rules
            and with NaN applied to pixels outside of the DEM valid bounds.
        """
        da = xr.open_dataarray(self.veg_base_path)
        da = da.squeeze(drop="band")

        da = self._reproject_match_to_dem(da)
        self._logger.info("Loaded initial vegetation raster")
        veg_type = da.to_numpy()

        self._logger.info("Subsetting initial vegetation raster to allowed types")
        # allowed veg types
        values_to_mask = [15, 16, 17, 18, 19, 20, 21, 22, 23, 26]
        # Create mask where True corresponds to values in the list
        type_mask = np.isin(veg_type, values_to_mask)
        veg_type = np.where(type_mask, veg_type, np.nan)

        # Mask the vegetation raster to only include valid DEM pixels
        self._logger.info("Masking vegetation raster to valid DEM pixels")
        dem_valid_mask = ~np.isnan(self.dem)
        veg_type = np.where(dem_valid_mask, veg_type, np.nan)

        if xarray:
            # Reassign np.array to origina DataArray. This ensure
            # the data is identical for either output type.
            da["veg_type_subset"] = (("y", "x"), veg_type)
            return da["veg_type_subset"]

        return veg_type

    def _load_veg_keys(self) -> pd.DataFrame:
        """load vegetation class names from database file"""
        dbf = gpd.read_file(self.veg_keys_path)
        # fix dtype
        dbf["Value"] = dbf["Value"].astype(int)
        self._logger.info("Loaded Vegetation Keys")
        return dbf

    def _get_salinity(self) -> np.ndarray:
        """Load salinity raster data (if available.)"""
        if self.salinity_path:
            # add loading code here
            self._logger.info("Loaded salinity from raster")
        else:
            self.salinity = hydro_logic.habitat_based_salinity(self.veg_type)
            self._logger.info("Creating salinity defaults from veg type array.")
            return self.salinity

    def _create_output_dirs(self):
        """Create an output location for state variables, model config,
        input data, and QC plots.

        (No logging because logger needs output location for log file first.)
        """
        output_dir_name = f"VegOut_{self.sim_start_time}"

        # Combine base directory and new directory name
        self.output_dir_path = os.path.join(self.output_base_dir, output_dir_name)
        # Create the directory if it does not exist
        os.makedirs(self.output_dir_path, exist_ok=True)

        # Create the 'run-input' subdirectory
        run_metadata_dir = os.path.join(self.output_dir_path, "run-metadata")
        os.makedirs(run_metadata_dir, exist_ok=True)

        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, run_metadata_dir)
        else:
            print("Config file not found at %s", self.config_path)

    def _save_state_vars(self, params: dict):
        """The method will save state variables after each timestep.

        This method should also include the config, input data, and QC plots.
        """
        template = self.water_depth.isel({"time": 0})  # subset to first month

        # veg type out
        filename_vegtype = utils.generate_filename(
            params=params,
            base_path=self.timestep_output_dir,
            parameter="VEGTYPE",
        )
        new_variables = {"veg_type": (self.veg_type, {"units": "veg_type"})}
        self.timestep_out = utils.create_dataset_from_template(template, new_variables)
        self.timestep_out["veg_type"].rio.to_raster(
            filename_vegtype.with_suffix(".tif")
        )

        # pct mast out
        # TODO: add perent mast handling

        # maturity out
        filename_maturity = utils.generate_filename(
            params=params,
            base_path=self.timestep_output_dir,
            parameter="MATURITY",
        )
        self.timestep_out["maturity"] = (("y", "x"), self.maturity)
        self.timestep_out["maturity"].rio.to_raster(
            filename_maturity.with_suffix(".tif")
        )

    def _create_timestep_dir(self, counter):
        """Create output directory for the current timestamp, where
        figures and output rasters will be saved.
        """
        self.timestep_output_dir = os.path.join(self.output_dir_path, counter)
        self.timestep_output_dir_figs = os.path.join(
            self.timestep_output_dir,
            "figs",
        )
        os.makedirs(self.timestep_output_dir, exist_ok=True)
        os.makedirs(self.timestep_output_dir_figs, exist_ok=True)

    def post_process(self):
        """After a run has been executed, this method generates a summary
        timeseries, and saves it as CSV in the "run-metadata" directory.
        """
        # uses utils.open_veg_multifile,
        # then utils.timeseries_outputs

        # options should be enabled/disabled here
        # TODO write WPU subset func: this will include
        # adding another data source, the WPU polygons, or
        # perhaps a raster layer that is coded with WPU number.
        # the code for that should also go in `utils`

        return NotImplementedError


class _TimestepFilter(logging.Filter):
    """A roundabout way to inject the current timestep into log records.
    Should & could be simplified.

    N/A if log messages occurs while self.current_timestep is not set.
    """

    def __init__(self, veg_transition_instance):
        super().__init__()
        self.veg_transition_instance = veg_transition_instance

    def filter(self, record):
        # Dynamically add the current timestep to log records
        record.timestep = (
            self.veg_transition_instance.current_timestep.strftime("%Y-%m-%d")
            if self.veg_transition_instance.current_timestep
            else "N/A"
        )
        return True
