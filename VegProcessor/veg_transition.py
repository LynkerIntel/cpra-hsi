import logging
import yaml
import subprocess
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import shutil
import re
import glob
import os
import gc
import copy
from typing import Optional
import rioxarray  # used for tif output
import rasterio.crs
from pathlib import Path

from datetime import datetime
import matplotlib.pyplot as plt

from output_vars import get_veg_variables

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
    where time series awareness is needed (i.e. subsetting WSE data). vegetation numpy
    arrays are dtype float32 by default, because int types have limited interoperability
    with the np.nan type, which is needed.


    Attributes: (only state variables listed)
    -----------
        self.veg_type
        self.maturity
        self.pct_mast_hard
        self.pct_mast_soft
        self.pct_no_mast
        self.salinity

    Methods
    -------
    step():
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
        self.wse_directory_path = self.config["raster_data"].get(
            "wse_directory_path"
        )
        self.wse_domain_path = self.config["raster_data"].get(
            "wse_domain_raster"
        )
        self.netcdf_hydro_path = self.config["raster_data"].get(
            "netcdf_hydro_path"
        )
        self.veg_base_path = self.config["raster_data"].get("veg_base_raster")
        self.veg_keys_path = self.config["raster_data"].get("veg_keys")
        self.salinity_path = self.config["raster_data"].get("salinity_raster")
        self.wpu_grid_path = self.config["raster_data"].get("wpu_grid")
        self.initial_maturity_path = self.config["raster_data"].get(
            "initial_maturity"
        )

        # polygon data
        self.wpu_polygons = self.config["polygon_data"].get("wpu_polygons")

        # simulation
        self.water_year_start = self.config["simulation"].get(
            "water_year_start"
        )
        self.water_year_end = self.config["simulation"].get("water_year_end")
        self.netcdf_hydro = self.config["simulation"].get("daily_hydro")
        self.years_mapping = self.config["simulation"].get("years_mapping")

        # metadata
        self.metadata = self.config["metadata"]
        self.scenario_type = self.config["metadata"].get(
            "scenario", ""
        )  # empty str if missing

        # output
        self.output_base_dir = self.config["output"].get("output_base")

        # Pretty-print the configuration
        config_pretty = yaml.dump(
            self.config, default_flow_style=False, sort_keys=False
        )

        # NetCDF data output
        sim_length = self.water_year_end - self.water_year_start

        self.file_params = {
            "model": self.metadata.get(
                "model"
            ),  # which model to run: one of VEG or HSI
            "hydro_source_model": self.metadata.get(
                "hydro_source_model"
            ),  # one of: HEC, MIK, or D3D
            "hydro_source_model_version": self.metadata.get(
                "hydro_source_model_version"
            ),  # model version, i.e. V1
            "water_year": "WY99",  # default for now, may be needed
            "sea_level_condition": self.metadata.get("sea_level_condition"),
            "flow_scenario": self.metadata.get("flow_scenario"),
            "input_group": self.metadata.get("input_group"),
            "output_group": self.metadata.get("output_group"),
            "wpu": "AB",
            "io_type": "O",
            "time_freq": "ANN",  # for annual output
            "year_range": (
                f"00_{str(sim_length + 1).zfill(2)}"
            ),  # 00 start (initial conditions)
            "output_version": self.metadata.get("output_version"),
        }

        # Generate filename early so it's available for logger and metadata files
        self.file_name = utils.generate_filename(
            params=self.file_params,
        )

        self._create_output_dirs()
        self.current_timestep = None
        self._setup_logger(log_level)
        self.timestep_output_dir = None

        # load sequence mapping (used for daily hydro data input)
        self.sequence_mapping = utils.load_sequence_csvs("./sequences/")

        # Log the configuration
        self._logger.info("Loaded Configuration:\n%s", config_pretty)
        self._get_git_commit_hash()

        self.dem = self._load_dem()
        self.hydro_domain = self._load_hecras_domain_raster()

        self.initial_veg_type = self._load_veg_initial_raster()  # static
        self.veg_type = self._load_veg_initial_raster()  # dynamic
        self.static_veg = self._load_veg_initial_raster(
            return_static_veg_only=True
        )
        self.veg_keys = self._load_veg_keys()

        self.maturity = self._load_initial_maturity_raster()
        self.wse = None
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

        self._create_output_file()

    def _setup_logger(self, log_level=logging.INFO):
        # always create a unique logger for each instance, using class name
        self._logger = logging.getLogger(
            f"{self.__class__.__name__}_{id(self)}"
        )
        self._logger.setLevel(log_level)

        # always remove old handlers (critical in Jupyter notebooks)
        if self._logger.hasHandlers():
            for handler in self._logger.handlers:
                self._logger.removeHandler(handler)
                handler.close()

        # now create fresh handlers
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        run_metadata_dir = os.path.join(self.output_dir_path, "run-metadata")
        os.makedirs(run_metadata_dir, exist_ok=True)
        log_file_path = os.path.join(
            run_metadata_dir, f"{self.file_name}_simulation.log"
        )
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [Timestep: %(timestep)s] - %(message)s"
        )

        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        self._logger.addHandler(ch)
        self._logger.addHandler(fh)

        # add the timestep filter
        filter_instance = _TimestepFilter(self)
        self._logger.addFilter(filter_instance)

        self._logger.info("Logger setup complete.")

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

    def step(self, timestep: pd.DatetimeTZDtype):
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
        self.wy = timestep.year

        self._logger.info("starting timestep: %s", timestep)
        self._create_timestep_dir(timestep)

        # copy existing veg types
        veg_type_in = self.veg_type.copy()

        # self.water_depth = self._load_stage_daily(self.wy)
        self.water_depth = self._load_stage_general(self.wy)

        # get salinity
        self.salinity = self._get_salinity()

        self.create_qc_arrays()

        # important: mask areas outside of domain before calculting transition:
        self._logger.info("Masking veg type array to domain.")
        self.veg_type = np.where(self.hydro_domain, self.veg_type, np.nan)

        # note: arrays are named for their starting veg type
        self.zone_v = veg_logic.zone_v(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            logger=self._logger,
        )
        self.zone_iv = veg_logic.zone_iv(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            logger=self._logger,
        )
        self.zone_iii = veg_logic.zone_iii(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            logger=self._logger,
        )
        self.zone_ii = veg_logic.zone_ii(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            logger=self._logger,
        )
        self.fresh_shrub = veg_logic.fresh_shrub(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            logger=self._logger,
        )
        self.fresh_marsh = veg_logic.fresh_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
            logger=self._logger,
        )
        self.intermediate_marsh = veg_logic.intermediate_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
            logger=self._logger,
        )
        self.brackish_marsh = veg_logic.brackish_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
            logger=self._logger,
        )
        self.saline_marsh = veg_logic.saline_marsh(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
            logger=self._logger,
        )
        self.water = veg_logic.water(
            self.veg_type,
            self.water_depth,
            self.timestep_output_dir_figs,
            self.salinity,
            logger=self._logger,
        )

        # stack partial update arrays for each zone
        # the final arrays is the static pixels that
        # are within the VegTransition domain, but
        # outside of the HEC-RAS domain.
        stacked_veg = np.stack(
            (
                self.zone_v["veg_type"],
                self.zone_iv["veg_type"],
                self.zone_iii["veg_type"],
                self.zone_ii["veg_type"],
                self.fresh_shrub["veg_type"],
                self.fresh_marsh["veg_type"],
                self.intermediate_marsh["veg_type"],
                self.brackish_marsh["veg_type"],
                self.saline_marsh["veg_type"],
                self.water["veg_type"],
                self.static_veg,
            )
        )

        if utils.has_overlapping_non_nan(stacked_veg):
            raise ValueError(
                "New vegetation stacked arrays cannot have overlapping values."
            )

        # combine arrays while preserving NaN
        self._logger.info("Combining new vegetation types into full array.")
        self.veg_type = np.full_like(self.zone_v["veg_type"], np.nan)
        for layer in stacked_veg:
            self.veg_type = np.where(
                np.isnan(self.veg_type), layer, self.veg_type
            )

        # plotting.np_arr(
        #     self.veg_type,
        #     title=f"All Types Output {self.current_timestep.strftime('%Y-%m-%d')} {self.scenario_type}",
        #     out_path=self.timestep_output_dir_figs,
        #     veg_palette=True,
        # )
        # plotting.sum_changes(
        #     veg_type_in,
        #     self.veg_type,
        #     plot_title=f"Timestep Veg Changes {self.current_timestep.strftime('%Y-%m-%d')} {self.scenario_type}",
        #     out_path=self.timestep_output_dir_figs,
        # )
        # # save water depth plots
        # plotting.water_depth(
        #     self.water_depth,
        #     out_path=self.timestep_output_dir_figs,
        #     wpu_polygons_path=self.wpu_polygons,
        # )

        self._calculate_maturity(veg_type_in)

        # serialize state variables: veg_type, maturity, mast %
        self._logger.info("saving state variables for timestep.")
        self._append_veg_vars_to_netcdf(timestep=self.current_timestep)

        self._logger.info("completed timestep: %s", timestep)
        self.current_timestep = None
        self.water_depth = None

        # clean up mpl objects
        plt.cla()
        plt.clf()
        plt.close("all")
        gc.collect()

    def run(self):
        """
        Run the vegetation transition model, with parameters defined in the configuration file.

        Start and end parameters are year, and handled as ints. No other frequency currently possible.
        """
        default_backend = plt.get_backend()

        try:
            # change to non-gui backend to prevent
            # memory leak if running in notebook
            plt.switch_backend("Agg")

            self._logger.info(
                "Starting simulation at %s. Period: %s - %s",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.water_year_start,
                self.water_year_end,
            )

            # plus 1 to make inclusive
            simulation_period = range(
                self.water_year_start, self.water_year_end + 1
            )
            self._logger.info(
                "Running model for: %s timesteps", len(simulation_period)
            )

            for i, wy in enumerate(simulation_period):
                self.step(
                    timestep=pd.to_datetime(f"{wy}-10-01"),
                )

            self._logger.info("Simulation complete")
            del self.water_depth
            logging.shutdown()

        finally:
            plt.switch_backend(default_backend)

    def _load_dem(self, cell: bool = False) -> np.ndarray:
        """Load project domain DEM.

        Params
        ------
        cell : bool
            If DEM should be downscaled to 480m cell size, default False

        Return
        ------
        domain : np.ndarray
            model domain in 60m or 480m resolution.
        """
        ds = xr.open_dataset(self.dem_path)
        ds = ds.squeeze(drop="band_data")
        da = ds.to_dataarray(dim="band")
        self._logger.info("Loaded DEM")

        if cell:
            da = da.coarsen(y=8, x=8, boundary="pad").mean()

        return da[0].to_numpy()

    def load_wse_wy(
        self,
        water_year: int,
        variable_name: str = "WSE_MEAN",
        date_format: str = "%Y_%m_%d",
        analog_sequence: bool = True,
    ) -> xr.Dataset:
        """DEPRECATED

        Load .tif files corresponding to a specific water year into an xarray.Dataset. This method
        uses lazy-loading via Dask.

        Each .tif file should have a filename that includes a variable name (e.g., `WSE_MEAN`)
        followed by a timestamp in the specified format (e.g., `2005_10_01`). The function will
        automatically extract the timestamps and assign them to a 'time' dimension in the resulting
        xarray.Dataset, but only for files within the specified water year.

        NOTE: Input WSE data (as of 2024-12-13) uses NaN to designate 0 depth. If this is false,
        because the input data has changed, or a different model with different NaN classification
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
            The name of the variable to use in the dataset (not the output var name).
        date_format : str, optional
            Format string for parsing dates from file names, default is "%Y_%m_%d".
            Adjust based on your file naming convention.
        analog_sequence : bool
            DEPRECATED. This is only for use when loading 25-year analog years sequences, where the
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
            glob.glob(
                os.path.join(self.wse_directory_path, "**/*.tif"),
                recursive=True,
            )
        )

        start_date = pd.to_datetime(f"{water_year - 1}-10-01")
        end_date = pd.to_datetime(f"{water_year}-09-30")

        selected_files = []
        time_stamps = []

        for f in tif_files:
            date_str = "_".join(os.path.basename(f).split("_")[2:5]).replace(
                ".tif", ""
            )
            file_date = pd.to_datetime(date_str, format=date_format)

            if start_date <= file_date <= end_date:
                selected_files.append(f)
                time_stamps.append(file_date)

        if not selected_files:
            self._logger.error(
                "No WSE files found for water year: %s", water_year
            )
            raise RuntimeError(
                f"No WSE files found for water year: {water_year}"
            )

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

        if analog_sequence:
            self._logger.info("Using sequence loading method.")
            new_timesteps = pd.date_range(
                f"{water_year-1}-10-01", f"{water_year}-09-01", freq="MS"
            )

            if not len(new_timesteps) == len(ds["time"]):
                raise ValueError("Timestep must be monthly.")

            # Replace the time coordinate with time from
            ds = ds.assign_coords(time=("time", new_timesteps))
            self._logger.info("Sequence timeseres updated to match filename.")

        # rename for internal consistency
        ds = ds.rename({"WSE_MEAN": "height"})
        self._logger.info(
            "Loaded HEC-RAS WSE Datset for water-year: %s", water_year
        )
        return ds

    def _load_stage_daily(self, water_year: int) -> xr.Dataset:
        """
        An alternative method to `load_wse_wy()`, to load water elevation data. This
        is designed to ingest stage data from HEC-RAS, as a daily NetCDF. Unlike
        `load_wse_wy()` it does not need a 25 year sequence generated by
        `utils.generate_combined_sequence()`, as it will load the correct analog year
        based on the sequence mappings. This method also includes the logic from
        `get_depth()` to correctly identify 0 depth and NaN, which are not distinguished
        in the raw model output. Finally, this data uses the `rio` `reproject_match()` method
        to ensure perfect overlap of the DEM and the hydro data.

        UNIT: Input in feet is converted to meters

        Parameters
        -----------
        water_year : int
            model timestep (as water year) used to select and load the
            corret analog year.

        Returns
        --------
        ds : xr.Dataset
            Dataset set of (time, x, y), with "height" var (i.e. depth)
        """
        self._logger.info(f"Loading hydro data with daily stage method.")
        quintile = self.sequence_mapping[water_year]
        analog_year = self.years_mapping[quintile]

        # build a 365-day date range by dropping Feb 29
        target_range = pd.date_range(
            f"{water_year-1}-10-01", f"{water_year}-09-30"
        )
        target_range = target_range[
            ~((target_range.month == 2) & (target_range.day == 29))
        ]

        nc_dir_path = os.path.join(
            self.netcdf_hydro_path,
            f"WY{analog_year}_{self.metadata['sea_level_condition']}_daily/netcdf4_**.nc",
        )
        self._logger.info("Loading files: %s", nc_dir_path)

        ds = xr.open_mfdataset(
            nc_dir_path,
            concat_dim="time",
            combine="nested",
            parallel=True,
            chunks={"time": 10},  # speedup (does not improve memory use)
            engine="h5netcdf",
        )

        # if analog has 366 timesteps (is leap)
        if ds.sizes["time"] == 366:
            # use filepath to get actual year
            match = re.search(r"WY(\d{4})", nc_dir_path)
            if match:
                actual_wy = int(match.group(1))
                # print(f"Extracted water year: {actual_wy}")
            else:
                raise ValueError("Water year not found in path.")

            # assign "actual" datetime to time dim, temporarily,
            # in order to drop feb 29
            full_range = pd.date_range(
                f"{actual_wy-1}-10-01", f"{actual_wy}-09-30"
            )
            ds = ds.assign_coords(time=("time", full_range))
            mask = ~((ds["time.month"] == 2) & (ds["time.day"] == 29))
            ds = ds.isel(time=mask)

        # lastly rename to match expected simulation dates, i.e. water year
        ds = ds.assign_coords(time=("time", target_range))
        ds = ds.rename({"Band1": "height"})

        # make crs visible to xarray/rio
        crs_obj = ds["transverse_mercator"].spatial_ref
        ds = ds.rio.write_crs(crs_obj)
        # reproject match to DEM
        ds = self._reproject_match_to_dem(ds)

        # fill zeros. This step is necessary to get 0 water depth from DEM and missing
        # WSE pixels, where missing data indicates "no inundation"
        ds = ds.fillna(0)
        # after filling zeros for areas with no inundation, apply domain mask,
        # so that areas outside of HECRAS domain are not classified as
        # dry (na is 0-filled above) when in fact that are outside of the domain.
        ds = ds.where(self.hydro_domain)

        self._logger.warning("Converting daily hydro: feet to meters")
        ds["height"] *= 0.3048  # UNIT: feet to meters
        return ds

    def _load_stage_general(self, water_year: int) -> xr.Dataset:
        """IN PROGRESS
        This is designed to ingest stage data from HEC-RAS, as an annual NetCDF.
        This function is called at the start of each timestep in the run loop.

        It is "general" because it is designed to work for all H&H model inputs:
        HECRAS, MIKE21, and Delf3D.

        Unlike `load_wse_wy()` it does not need a 25 year sequence generated by
        `utils.generate_combined_sequence()`, as it will load the correct analog year
        based on the sequence mappings. This method also includes the logic from
        `get_depth()` to correctly identify 0 depth and NaN, which are not distinguished
        in the raw model output. Finally, this data uses the `rio` `reproject_match()` method
        to ensure perfect overlap of the DEM and the hydro data.

        UNIT: Input in feet is converted to meters???

        Parameters
        -----------
        water_year : int
            model timestep (as water year) used to select and load the
            corret analog year.

        Returns
        --------
        ds : xr.Dataset
            Dataset set of (time, x, y), with "height" var (i.e. depth), where
            time is a single analog year, with timestamps updated to match the
            model timestep.
        """
        self._logger.info(
            f"Loading hydro data with universal daily stage method."
        )
        quintile = self.sequence_mapping[water_year]
        analog_year = self.years_mapping[quintile]

        # build a 365-day date range by dropping Feb 29
        target_range = pd.date_range(
            f"{water_year-1}-10-01", f"{water_year}-09-30"
        )
        target_range = target_range[
            ~((target_range.month == 2) & (target_range.day == 29))
        ]

        nc_path = os.path.join(
            self.netcdf_hydro_path,
            f"AMP_{self.file_params['hydro_source_model']}_WY{analog_year}_"
            f"{self.metadata['sea_level_condition']}_X_99_99_DLY_"
            f"{self.file_params['input_group']}_AB_O_STAGE_"
            f"{self.file_params['hydro_source_model_version']}.nc",
        )
        self._logger.info("Loading files: %s", nc_path)

        ds = xr.open_dataset(
            nc_path,
            engine="h5netcdf",
        )

        # if analog has 366 timesteps (is leap)
        if ds.sizes["time"] == 366:
            # use filepath to get actual year
            match = re.search(r"WY(\d{4})", nc_path)
            if match:
                actual_wy = int(match.group(1))
                # print(f"Extracted water year: {actual_wy}")
            else:
                raise ValueError("Water year not found in path.")

            # assign "actual" datetime to time dim, temporarily,
            # in order to drop feb 29
            full_range = pd.date_range(
                f"{actual_wy-1}-10-01", f"{actual_wy}-09-30"
            )
            ds = ds.assign_coords(time=("time", full_range))
            mask = ~((ds["time.month"] == 2) & (ds["time.day"] == 29))
            ds = ds.isel(time=mask)

        # rename to match expected simulation dates, i.e. water year
        ds = ds.assign_coords(time=("time", target_range))

        # model specific var names:
        if self.file_params["hydro_source_model"] == "HEC":
            ds = ds.rename({"Band1": "height"})
        if self.file_params["hydro_source_model"] == "D3D":
            ds = ds.rename({"waterlevel": "height"})
        # extract height var as da
        height_da = ds["height"]

        # handle varied CRS metadata locations between model files
        if self.file_params["hydro_source_model"] == "D3D":
            # D3D: Get CRS from crs variable's crs_wkt attribute
            crs_wkt = ds["crs"].attrs.get("crs_wkt")
            height_da = height_da.rio.write_crs(crs_wkt)
        elif "transverse_mercator" in ds:
            # HEC-RAS: Get CRS from transverse_mercator variable's spatial_ref attribute
            crs_wkt = ds["transverse_mercator"].attrs.get("spatial_ref")
            height_da = height_da.rio.write_crs(crs_wkt)

        height_da = self._reproject_match_to_dem(height_da)
        # new dataset with reprojected height
        ds = xr.Dataset({"height": height_da})

        # fill zeros. This step is necessary to get 0 water depth from DEM and missing
        # WSE pixels, where missing data indicates "no inundation"
        ds = ds.fillna(0)
        # after filling zeros for areas with no inundation, apply domain mask,
        # so that areas outside of HECRAS domain are not classified as
        # dry (na is 0-filled above) when in fact that are outside of the domain.
        ds = ds.where(self.hydro_domain)

        # self._logger.warning("Converting daily hydro: feet to meters")
        # ds["height"] *= 0.3048  # UNIT: feet to meters
        return ds

    def _reproject_match_to_dem(
        self, ds: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        """
        Temporary fix to match WSE model output to 60m DEM grid.
        TODO: use existing DEM to save time (no need to read every time)
        """
        ds_dem = xr.open_dataset(self.dem_path)
        da_dem = ds_dem.squeeze(drop="band_data").to_dataarray(dim="band")
        ds_reprojected = ds.rio.reproject_match(da_dem)
        ds_dem.close()
        da_dem.close()
        return ds_reprojected

    def _get_depth(self) -> xr.Dataset:
        """Calculate water depth from DEM and Water Surface Elevation and subset
        difference array to valid HECRAS domain. Results is subset to valid HECRAS
        pixels before being returned.

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
        # fill zeros. This step is necessary to get 0 water depth from DEM and missing
        # WSE pixels, where missing data indicates "no inundation"
        ds = ds.fillna(0)

        # after filling zeros for areas with no inundation, apply domain mask,
        # so that areas outside of HECRAS domain are not classified as
        # dry (na is 0-filled above) when in fact that are outside of the domain.
        ds = ds.where(self.hydro_domain)
        return ds

    def _calculate_maturity(self, veg_type_in: np.ndarray):
        """
        +1 year maturity for pixels without vegetation changes.

        TODO: Should static veg pixel increment age? Or should only valid WSE pixels
        advance?

        Parameters
        ----------
        veg_type_in : np.ndarray
            veg_type array at start of timestep, before transition calculations.
        """
        # Ensure both arrays have the same shape
        if veg_type_in.shape != self.veg_type.shape:
            raise ValueError(
                "Timestep input and output array have different shapes!"
            )

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
            self._logger.warning(
                "No forested pixels had maturity increment increase."
            )

        if not combined_mask_change.any():
            self._logger.warning(
                "No forested pixels changed in prior timestep."
            )

        if utils.common_true_locations(
            np.stack((combined_mask_change, combined_mask_no_change))
        ):
            raise ValueError(
                "Forested types have overlapping True location(s)"
            )

        # if forested pixels change, reset age to 0
        self.maturity[combined_mask_change] = 0
        self._logger.info("Maturity reset for changed veg type (forested)")
        # if forested pixels are the same, add one year
        self.maturity[combined_mask_no_change] += 1
        self._logger.info(
            "Maturity incremented for unchanged veg types (forested)"
        )

        # all other types (non-forested, non-handled) to np.nan
        self.maturity[~type_mask] = np.nan

        # plotting.np_arr(
        #     self.maturity,
        #     title=f"Timestep Maturity {self.current_timestep.strftime('%Y-%m-%d')} {self.scenario_type}",
        #     out_path=self.timestep_output_dir_figs,
        # )

    def _load_veg_initial_raster(
        self,
        xarray: bool = False,
        all_types: bool = False,
        return_static_veg_only: bool = False,
    ) -> np.ndarray | xr.Dataset:
        """This method will load the base veg raster, from which the model will iterate forwards,
        according to the transition logic. The veg types are subset to the DEM boundary before
        being returned.

        Parameters
        ----------
        xarray : bool
            True if xarray output format is needed. Default is False.
        all_types : bool
            True if all veg types are required, False if only transitioning veg types
            are needed. Default is False.

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

        # allowed veg types
        if not all_types:
            self._logger.info(
                "Subsetting initial vegetation raster to allowed types"
            )
            values = [15, 16, 17, 18, 19, 20, 21, 22, 23, 26]
            # Create mask where True corresponds to values in the list
            type_mask = np.isin(veg_type, values)
            veg_type = np.where(type_mask, veg_type, np.nan)

        # Mask the vegetation raster to only include valid DEM pixels
        self._logger.info("Masking vegetation raster to valid DEM pixels.")
        dem_valid_mask = ~np.isnan(self.dem)

        if return_static_veg_only:
            self._logger.info(
                "Returning array of static vegetation pixels only."
            )
            # combine DEM valid mask and inverse of HEC-RAS domain valid mask
            veg_type = np.where(
                dem_valid_mask & ~self.hydro_domain, veg_type, np.nan
            )

        else:
            # only apply valid DEM mask
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

    def _load_initial_maturity_raster(self) -> np.ndarray:
        """Load initial conditions for vegetation maturity."""
        self._logger.info("Loading initial maturity raster.")
        da = xr.open_dataarray(self.initial_maturity_path)
        da = da["band" == 0]
        da = self._reproject_match_to_dem(da)
        return da.to_numpy()

    def _load_hecras_domain_raster(self, cell: bool = False) -> np.ndarray:
        """Load raster file specifying the boundary of the HECRAS domain.

        Params
        -------
        cell : bool
            True if array should be downscaled to 480m "cell" resolution. If True,
            the array is returned as "data" with NaN. If False, data is returned as
            boolean (the format expected by `VegTransition` methods).
        """
        # load raster
        self._logger.info("Loading WSE domain extent raster.")
        da = xr.open_dataarray(self.wse_domain_path)
        da = da.squeeze(drop="band")
        # reproject match to DEM
        da = self._reproject_match_to_dem(da)

        if cell:
            da = da.coarsen(y=8, x=8, boundary="pad").mean()
            return da.to_numpy()

        da = da.fillna(0)  # fill 0 so that .astype(bool) does not fail
        da = da.astype(bool)
        return da.to_numpy()

    def _get_salinity(self) -> np.ndarray:
        """Load salinity raster data (if available.)"""
        if self.salinity_path:
            # add loading code here
            self._logger.info("Loaded salinity from raster")
        else:
            self.salinity = hydro_logic.habitat_based_salinity(
                veg_type=self.veg_type, domain=self.hydro_domain
            )
            self._logger.info(
                "Creating salinity defaults from veg type array."
            )
            return self.salinity

    def _create_output_dirs(self):
        """Create an output location for state variables, model config,
        input data, and QC plots. And copy config file into output
        location.

        (No logging because logger needs output location for log file first.)
        """
        naming_convention = utils.generate_filename(
            params=self.file_params,
            # base_path=self.timestep_output_dir,
            # parameter="DATA",
        )
        # create output file before NetCDF, so that
        # metadatafiles can use it.
        # self.file_name = utils.generate_filename(
        #     params=self.file_params,
        #     # base_path=self.timestep_output_dir,
        #     # parameter="DATA",
        # )

        output_dir_name = naming_convention

        # Combine base directory and new directory name
        self.output_dir_path = os.path.join(
            self.output_base_dir, output_dir_name
        )
        # Create the directory if it does not exist
        os.makedirs(self.output_dir_path, exist_ok=True)

        # create the 'run-metadata' subdirectory
        self.run_metadata_dir = os.path.join(
            self.output_dir_path, "run-metadata"
        )
        os.makedirs(self.run_metadata_dir, exist_ok=True)

        # create the 'figs' subdirectory
        self.run_figs_dir = os.path.join(self.run_metadata_dir, "figs")
        os.makedirs(self.run_figs_dir, exist_ok=True)

        if os.path.exists(self.config_path):
            config_filename = os.path.basename(self.config_path)
            config_name, config_ext = os.path.splitext(config_filename)
            new_config_path = os.path.join(
                self.run_metadata_dir,
                f"{self.file_name}_{config_name}{config_ext}",
            )
            shutil.copy(self.config_path, new_config_path)
        else:
            print("Config file not found at %s", self.config_path)

    def _create_output_file(self):
        """VegTransition: Create NetCDF file for data output.

        Returns
        -------
        None
        """

        self.netcdf_filepath = os.path.join(
            self.output_dir_path, f"{self.file_name}.nc"
        )

        # load DEM, use coords
        da = xr.open_dataarray(self.dem_path)
        da = da["band" == 0]
        x = da.coords["x"].values
        y = da.coords["y"].values

        # Define the new time coordinate
        time_range = pd.date_range(
            # start year minus 1 for initial conditions year
            start=f"{self.water_year_start - 1}-10-01",
            end=f"{self.water_year_end}-10-01",
            freq="YS-OCT",
        )  # Annual start

        # encoding defined here and at append
        encoding = {
            "time": {
                "units": "days since 1850-01-01T00:00:00",
                "calendar": "gregorian",
                "dtype": "float64",
            }
        }

        ds = xr.Dataset(
            {  # init with arrays that have t=0 values
                "veg_type": (
                    ["time", "y", "x"],
                    np.full(
                        (len(time_range), *self.dem.shape),
                        np.nan,
                        dtype=np.float32,
                    ),
                    {
                        "grid_mapping": "spatial_ref",  # Link CRS variable
                        "units": "unitless",
                        "long_name": "veg type",
                    },
                ),
                "maturity": (
                    ["time", "y", "x"],
                    np.full(
                        (len(time_range), *self.dem.shape),
                        np.nan,
                        dtype=np.float32,
                    ),
                    {
                        "grid_mapping": "spatial_ref",
                        "units": "years",
                        "long_name": "forested vegetation age",
                    },
                ),
            },
            coords={
                "x": (
                    "x",
                    x,
                    {
                        "units": "m",
                        "long_name": "Easting",
                        "standard_name": "projection_x_coordinate",
                    },
                ),
                "y": (
                    "y",
                    y,
                    {
                        "units": "m",
                        "long_name": "Northing",
                        "standard_name": "projection_y_coordinate",
                    },
                ),
                "time": (
                    "time",
                    time_range,
                    {
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
            },
            attrs={"title": "VEG"},
        )

        # add initial conditions to first timestep
        self._logger.info(
            "Addining initial vegetation conditions to timestep zero."
        )
        ds["veg_type"].loc[{"time": time_range[0]}] = (
            self.initial_veg_type.astype(np.float32)
        )
        self._logger.info(
            "Addining initial maturity conditions to timestep zero."
        )
        ds["maturity"].loc[{"time": time_range[0]}] = self.maturity.astype(
            np.float32
        )

        ds = ds.rio.write_crs("EPSG:6344")
        # Save dataset to NetCDF with explicit encoding
        ds.to_netcdf(self.netcdf_filepath, encoding=encoding)
        ds.close()
        da.close()
        self._logger.info("Initialized NetCDF file: %s", self.netcdf_filepath)

    def _append_veg_vars_to_netcdf(
        self, timestep: pd.DatetimeTZDtype, variables_to_append=None
    ):
        """
        Append timestep data to the NetCDF file for all vegetation variables.

        This method opens the NetCDF file, updates or initializes the following variables
        for the current timestep, and writes the changes back to disk.

        Vegetation variables are stored in a dictionary `veg_variables` with the following structure:

            {
                <variable_name>: [<data_array>, <data_type>, <nc_attributes_dict>],
                ...
            }

        Where:
            <variable_name>: a string representing the name of the variable (e.g., "veg_type", "maturity", "qc_annual_mean_salinity", etc.).
            <data_array>: the corresponding data array for the variable at the current timestep.
            <data_type>: the NumPy data type (e.g., np.float32 or bool).
            <nc_attributes_dict>: a dictionary containing attributes for the NetCDF variable, typically including:
                - "grid_mapping": a string linking the variable to its coordinate reference system (e.g., "crs"),
                - "units": a string indicating the measurement units (e.g., "unitless", "years", "meters", etc.),
                - "long_name": a descriptive name for the variable,
                - "description": (optional) additional details about the variable.

        Parameters
        ----------
        timestep : pd.DatetimeTZDtype
            Pandas datetime object representing the current timestep.

        variables_to_append : list
            list of arrays to save. Necessary to save intermediate output
            during a timestep.

        Returns
        -------
        None

        Notes
        -----
        - The method assumes that the time coordinate in the NetCDF file is formatted as YYYY-MM-DD.
        - If a variable does not exist in the file, it is initialized with a default value (NaN for floats and False for booleans).
        - Boolean data arrays are converted to boolean type after replacing NaN with False.
        - The dataset is saved in append mode before closing it.
        """
        encoding = {
            "time": {
                "units": "days since 1850-01-01T00:00:00",
                "calendar": "gregorian",
                "dtype": "float64",
            }
        }

        timestep_str = timestep.strftime("%Y-%m-%d")

        # Set default variables to append if not supplied, excluding all QC variables
        if variables_to_append is None:
            variables_to_append = ["veg_type", "maturity"]

        with xr.open_dataset(self.netcdf_filepath, cache=False) as ds:
            ds_loaded = ds.load()  # loads into memory and closes file

        # with xr.open_dataset(self.netcdf_filepath, cache=False) as ds:
        veg_variables = get_veg_variables(self)

        for var_name, (data, dtype, nc_attrs) in veg_variables.items():
            if var_name not in variables_to_append:
                continue
            # Check if the variable exists in the dataset, if not, initialize it
            if var_name not in ds_loaded:
                shape = (
                    len(ds_loaded.time),
                    len(ds_loaded.y),
                    len(ds_loaded.x),
                )
                default_value = False if dtype == bool else np.nan
                ds_loaded[var_name] = (
                    ["time", "y", "x"],
                    np.full(shape, default_value, dtype=dtype),
                    nc_attrs,
                )

            # Handle 'condition' variables (booleans)
            if dtype == bool:
                data = np.nan_to_num(data, nan=False).astype(bool)

            # Assign the data to the dataset for the specific time step
            ds_loaded[var_name].loc[{"time": timestep}] = data.astype(
                ds_loaded[var_name].dtype
            )

        # ds.close()
        ds_loaded.to_netcdf(
            self.netcdf_filepath,
            mode="a",
            engine="h5netcdf",
            encoding=encoding,
        )
        ds_loaded.close()
        self._logger.info("Appended timestep %s to NetCDF file.", timestep_str)

    def _create_timestep_dir(self, date: pd.DatetimeTZDtype):
        """Create output directory for the current timestamp, where
        figures and output rasters will be saved.

        Parameters
        ----------
        timestep : pd.DatetimeTZDtype
            Pandas datetime object of current timestep.

        Returns
        -------
        None
        """

        self.timestep_output_dir_figs = os.path.join(
            self.run_figs_dir, f"{date.strftime('%Y%m%d')}"
        )
        # self.timestep_output_dir_figs = os.path.join(
        #     self.timestep_output_dir,
        #     "figs",
        # )
        # os.makedirs(self.timestep_output_dir, exist_ok=True)
        os.makedirs(self.timestep_output_dir_figs, exist_ok=True)

    # def save_water_depth(self, params: dict):
    #     """
    #     Output water depth (calculated from difference of DEM and WSE).
    #     This method us unique from `save_state_vars` as water depth is
    #     (1) a model input (not state var), and (2) an xr.Dataset. This
    #     method is designed to work for either monthly or daily frequency.
    #     """
    #     # create dir for monthly or daily water depth files

    #     # create filenames
    #     dates = self.water_depth.time.values

    #     filenames = []
    #     for d in dates:
    #         fn = utils.generate_filename(
    #             params=params,
    #             base_path=self.timestep_output_dir,
    #             parameter="WATER_DEPTH",
    #         )
    #         filenames.append(fn)

    #     # for each timestep, output TIF? or do they want a figure?

    def post_process(self):
        """After a run has been executed, this method generates a summary
        timeseries, and saves it as CSV in the "run-metadata" directory.

        TODO: `utils.pixel_sums_full_domain` is very slow. It takes nearly
            8 minutes. Speeding this up would nearly half the total
            exection time.
        """
        logging.info("Running post-processing routine.")
        wpu = xr.open_dataarray(self.wpu_grid_path, engine="rasterio")
        wpu = wpu["band" == 0]
        # Replace 0 with NaN (Zone 0 is outside of all WPU polygons)
        wpu = xr.where(wpu != 0, wpu, np.nan)

        ds = xr.open_dataset(self.netcdf_filepath)
        # ds = utils.open_veg_multifile(self.output_dir_path)

        logging.info("Calculating WPU veg type sums.")
        df = utils.wpu_sums(ds_veg=ds, zones=wpu)

        # rename cols from int to type name
        # types_dict = dict(self.veg_keys[["Value", "Class"]].values)
        # df.rename(columns=types_dict, inplace=True)

        outpath = os.path.join(
            self.run_metadata_dir,
            f"{self.file_name}_wpu_vegtype_timeseries.csv",
        )
        df.to_csv(outpath)

        logging.info("Calculating full-domain veg type sums.")

        df_full_domain = utils.pixel_sums_full_domain(ds=ds)
        outpath = os.path.join(
            self.run_metadata_dir, f"{self.file_name}_vegtype_timeseries.csv"
        )
        df_full_domain.to_csv(outpath)

        logging.info("Creating variable name text file")
        outpath = os.path.join(
            self.run_metadata_dir, f"{self.file_name}_veg_netcdf_variables.csv"
        )
        attrs_df = utils.dataset_attrs_to_df(
            ds,
            selected_attrs=[
                "long_name",
                "description",
                "units",
            ],
        )
        attrs_df.to_csv(outpath, index=False)

        logging.info("Post-processing complete.")

    def create_qc_arrays(self):
        """
        Create QC arrays with variables defined by JV, used to ensure
        vegetation transition ruleset is working as intended. Arrays are serialized
        to NetCDF then deleted and no longer accessible to the class instance.
        """
        self._logger.info("Creating QA/QC arrays.")
        self.qc_annual_mean_salinity = utils.qc_annual_mean_salinity(
            self.salinity,
        )
        self.qc_annual_inundation_depth = utils.qc_annual_inundation_depth(
            self.water_depth
        )
        self.qc_annual_inundation_duration = (
            utils.qc_annual_inundation_duration(self.water_depth)
        )
        self.qc_growing_season_depth = utils.qc_growing_season_depth(
            self.water_depth,
        )
        self.qc_growing_season_inundation = utils.qc_growing_season_inundation(
            self.water_depth
        )
        self.qc_tree_establishment_bool = utils.qc_tree_establishment_bool(
            self.water_depth
        )
        (
            self.qc_march_water_depth,
            self.qc_april_water_depth,
            self.qc_may_water_depth,
            self.qc_june_water_depth,
        ) = utils.qc_tree_establishment_info(self.water_depth)

        self._logger.info("Saving QC arrays to NetCDF.")
        self._append_veg_vars_to_netcdf(
            timestep=self.current_timestep,
            variables_to_append=[
                "qc_annual_mean_salinity",
                "qc_annual_inundation_depth",
                "qc_annual_inundation_duration",
                "qc_growing_season_depth",
                "qc_growing_season_inundation",
                "qc_tree_establishment_bool",
                "qc_march_water_depth",
                "qc_april_water_depth",
                "qc_may_water_depth",
                "qc_june_water_depth",
            ],
        )
        # Delete QC arrays after saving to reduce memory usage
        self.qc_annual_mean_salinity = None
        self.qc_annual_inundation_depth = None
        self.qc_annual_inundation_duration = None
        self.qc_growing_season_depth = None
        self.qc_growing_season_inundation = None
        self.qc_march_water_depth = None
        self.qc_april_water_depth = None
        self.qc_may_water_depth = None
        self.qc_june_water_depth = None
        gc.collect()


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
