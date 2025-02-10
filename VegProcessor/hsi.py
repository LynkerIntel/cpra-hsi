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

# import veg_logic
import hydro_logic
import plotting
import utils

import veg_transition as vt
from species_hsi import alligator, crawfish, baldeagle


# this is a c/p from veg class, not sure why I need it again here.
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


class HSI(vt.VegTransition):
    """HSI model framework."""

    def __init__(self, config_file: str, log_level: int = logging.INFO):
        """
        Initialize by setting up logger, loading resource paths, and creating empty
        arrays for state variables.

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
        self.veg_type_path = self.config["raster_data"].get("veg_type_path")
        self.veg_keys_path = self.config["raster_data"].get("veg_keys")
        self.salinity_path = self.config["raster_data"].get("salinity_raster")

        self.flotant_marsh_path = self.config["raster_data"].get("flotant_marsh_raster")
        # self.flotant_marsh_keys_path = self.config["raster_data"].get("flotant_marsh_keys")

        # simulation
        self.water_year_start = self.config["simulation"].get("water_year_start")
        self.water_year_end = self.config["simulation"].get("water_year_end")
        self.run_hsi = self.config["simulation"].get("run_hsi")
        self.analog_sequence = self.config["simulation"].get("wse_sequence_input")

        # metadata
        self.metadata = self.config["metadata"]
        # self.scenario_type = self.config["metadata"].get(
        #     "scenario", ""
        # )  # empty str if missing

        # output
        self.output_base_dir = self.config["output"].get("output_base")

        # Pretty-print the configuration
        config_pretty = yaml.dump(
            self.config,
            default_flow_style=False,
            sort_keys=False,
        )

        self._create_output_dirs()
        self.current_timestep = None  # set in step() method
        self._setup_logger(log_level)
        self.timestep_output_dir = None  # set in step() method

        # Pretty-print the configuration
        config_pretty = yaml.dump(
            self.config,
            default_flow_style=False,
            sort_keys=False,
        )

        # Log the configuration
        self._logger.info("Loaded Configuration:\n%s", config_pretty)

        # Generate static variables
        self.dem = self._load_dem()
        self.veg_keys = self._load_veg_keys()
        self.edge = self._calculate_edge()
        self.initial_veg_type = self._load_veg_initial_raster(
            xarray=True,
            all_types=True,
        )
        self.flotant_marsh = self._calculate_flotant_marsh()

        # Get pct cover for prevously defined static variables
        self._calculate_pct_cover_static()

        # Dynamic Variables
        self.wse = None
        self.maturity = np.zeros_like(self.dem)
        self.water_depth_annual_mean = None
        self.veg_ts_out = None  # xarray output for timestep
        self.water_depth_monthly_mean_jan_aug = None
        self.water_depth_monthly_mean_sept_dec = None

        # HSI models
        self.alligator = None
        self.crawfish = None
        self.baldeagle = None
        # self.blackbear = None

        # datasets
        self.pct_cover_veg = None

        # HSI Variables
        self.pct_open_water = None
        self.avg_water_depth_rlt_marsh_surface = None
        self.mean_annual_salinity = None

        self.pct_swamp_bottom_hardwood = None
        self.pct_fresh_marsh = None
        self.pct_intermediate_marsh = None
        self.pct_brackish_marsh = None
        self.pct_saline_marsh = None

        self.pct_zone_v = None
        self.pct_zone_iv = None
        self.pct_zone_iii = None
        self.pct_zone_ii = None
        self.pct_fresh_shrubs = None

        self.pct_bare_ground = None
        self.pct_dev_upland = None  # does not change
        # self.pct_dev_upland = self._calculate_pct_cover_static()

        # NetCDF data output
        sim_length = self.water_year_end - self.water_year_start

        file_params = {
            "model": self.metadata.get("model"),
            "scenario": self.metadata.get("scenario"),
            "group": self.metadata.get("group"),  # not sure if this is correct,
            "wpu": "AB",
            "io_type": "O",
            "time_freq": "ANN",  # for annual output
            "year_range": f"01_{str(sim_length + 1).zfill(2)}",
            # "parameter": "NA",  # ?
        }

        self._create_output_file(file_params)

    def _setup_logger(self, log_level=logging.INFO):
        """Set up the logger for the VegTransition class."""
        self._logger = logging.getLogger("HSI")
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

    def step(self, date: pd.DatetimeTZDtype):
        """Calculate Indices & Advance the HSI models by one step.

        TODO: for memory efficiency, 60m arrays should be garbage collected after creation of
        480m HSI input arrays.
        """
        self.current_timestep = date  # Set the current timestep
        wy = date.year

        self._logger.info("starting timestep: %s", date)
        self._create_timestep_dir(date)

        # calculate depth
        # avg_water_depth_rlt_marsh_surface
        self.wse = self.load_wse_wy(wy, variable_name="WSE_MEAN")
        self.wse = self._reproject_match_to_dem(self.wse)  # TEMPFIX
        self.water_depth_annual_mean = self._get_water_depth_annual_mean()
        # self.water_depth_monthly_mean_jan_aug = self._get_water_depth_monthly_mean_jan_aug()
        # self.water_depth_monthly_mean_sept_dec = self._get_water_depth_monthly_mean_sept_dec()

        # load veg type
        self.veg_type = self._load_veg_type()

        # calculate pct cover for all veg types
        self._calculate_pct_cover()
        self.mean_annual_salinity = hydro_logic.habitat_based_salinity(self.veg_type)

        # bald eagle

        # SAVE ALL SI INDICES INDIVIDUALLY

        # run HSI models for timestep
        if self.run_hsi:

            self.alligator = alligator.AlligatorHSI.from_hsi(self)
            self.crawfish = crawfish.CrawfishHSI.from_hsi(self)
            self.baldeagle = baldeagle.BaldEagleHSI.from_hsi(self)
            # self.black_bear = BlackBearHSI(self)

            self._append_hsi_vars_to_netcdf(timestep=self.current_timestep)

        self._logger.info("completed timestep: %s", date)
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

        for wy in simulation_period:
            self.step(pd.to_datetime(f"{wy}-10-01"))

        self._logger.info("Simulation complete")
        logging.shutdown()

    def _load_veg_type(self) -> xr.DataArray:
        """Load VegTransition output raster data.

        Returns : xr.DataArray
            a single WY, defined by `self.current_timestep`.
        """
        logging.info("Loading vegetation data.")
        file_path = os.path.join(self.veg_type_path, "data_output.nc")
        time_str = self.current_timestep.strftime("%Y%m%d")
        ds = xr.open_dataset(file_path)
        da = ds.sel({"time": time_str})["veg_type"]
        return da

    def _calculate_pct_cover(self):
        """Get percent coverage for each 480m cell, based on 60m veg type pixels. This
        function is called for every HSI timestep.

        Derived from VegTransition Output
        """
        # logical index for coarsening (i.e # of pixels)
        # this generates ds with all veg types.
        # x, y, dims -> 480 / 60 = 8
        ds = utils.generate_pct_cover(
            data_array=self.veg_type,
            veg_keys=self.veg_keys,
            x=8,
            y=8,
            boundary="pad",
        )

        ds_swamp_blh = utils.generate_pct_cover_custom(
            data_array=self.veg_type,
            veg_types=[15, 16, 17, 18],  # these are the BLH zones + swamp
            x=8,
            y=8,
            boundary="pad",  # not needed for partial pixels, fyi
        )

        # VEG TYPES AND THIER MAPPED NUMBERS FROM
        # 2  Developed High Intensity
        # 3  Developed Medium Intensity
        # 4  Developed Low Intensity
        # 5  Developed Open Space
        # 6  Cultivated Crops
        # 7  Pasture/Hay
        # 8  Grassland/Herbaceous
        # 9  Upland - Mixed Deciduous Forest
        # 10  Upland - Mixed Evergreen Forest
        # 11  Upland Mixed Forest
        # 12  Upland Scrub/Shrub
        # 13  Unconsolidated Shore
        # 14  Bare Land
        # 15  Zone V (Upper BLH)
        # 16  Zone IV (Middle BLH)
        # 17  Zone III (Lower BLH)
        # 18  Zone II (Swamp)
        # 19  Fresh Shrubs
        # 20  Fresh Marsh
        # 21  Intermediate Marsh
        # 22  Brackish Marsh
        # 23  Saline Marsh
        # 24  Palustrine Aquatic Bed
        # 25  Estuarine Aquatic Bed
        # 26  Open Water

        # self._logger.debug("Generated ds_swamp_blh: %s", ds_swamp_blh)

        self.pct_bare_ground = ds["pct_cover_14"].to_numpy()
        self.pct_zone_v = ds["pct_cover_15"].to_numpy()
        self.pct_zone_iv = ds["pct_cover_16"].to_numpy()
        self.pct_zone_iii = ds["pct_cover_17"].to_numpy()
        self.pct_zone_ii = ds["pct_cover_18"].to_numpy()
        self.pct_fresh_shrubs = ds["pct_cover_19"].to_numpy()

        self.pct_fresh_marsh = ds["pct_cover_20"].to_numpy()
        self.pct_intermediate_marsh = ds["pct_cover_21"].to_numpy()
        self.pct_brackish_marsh = ds["pct_cover_22"].to_numpy()
        self.pct_saline_marsh = ds["pct_cover_23"].to_numpy()
        self.pct_open_water = ds["pct_cover_26"].to_numpy()

        # Zone V, IV, III
        # self.pct_swamp_bottom_hardwood = ds_blh.to_numpy()

        # Zone V, IV, III, (BLH's) II (swamp)
        self.pct_swamp_bottom_hardwood = ds_swamp_blh.to_numpy()
        # self._logger.debug("Completed _calculate_pct_cover")

    def _calculate_pct_cover_static(self):
        """Get percent coverage for each 480m cell, based on 60m veg type pixels.
        This method is called during initialization, for static variables.
        """
        ds_dev_upland = utils.generate_pct_cover_custom(
            data_array=self.initial_veg_type,
            # these are the dev'd (4) and upland (4)
            veg_types=[2, 3, 4, 5, 9, 10, 11, 12],
            x=8,
            y=8,
            boundary="pad",
        )

        # Developed Land (4 diff types) and Upland (also 4)
        self.pct_dev_upland = ds_dev_upland.to_numpy()

        # Flotant marsh for baldeagle
        self.pct_flotant_marsh = utils.coarsen_and_reduce(
            da=self.flotant_marsh,
            veg_type=True,
            x=8,
            y=8,
            boundary="pad",
        ).to_numpy()

    def _calculate_edge(self) -> np.ndarray:
        """
        Calculate percent of 480m cell that is marsh edge.
        """
        # Load veg base and use as template to create arrays for the main state variables
        initial_veg = self._load_veg_initial_raster(xarray=True)

        logging.info("Calculating water edge pixels.")
        open_water = 26
        # can't use np.nan (float), need to use an int
        fill_value = -99

        # Check neighbors
        neighbors = [
            initial_veg.shift(x=1, fill_value=fill_value),  # left
            initial_veg.shift(x=-1, fill_value=fill_value),  # right
            initial_veg.shift(y=1, fill_value=fill_value),  # down
            initial_veg.shift(y=-1, fill_value=fill_value),  # up
        ]

        # Combine all neighbor comparisons
        edge_pixels = xr.concat(
            [(neighbor == open_water) for neighbor in neighbors], dim="direction"
        ).any(dim="direction")

        logging.info("Calculating percent of water edge pixels in cell.")
        # get pct of cell that is edge
        # run coarsen w/ True as valid veg type
        da = utils.coarsen_and_reduce(
            da=edge_pixels,
            veg_type=True,
            x=8,
            y=8,
            boundary="pad",
        )
        return da.to_numpy()

    def _calculate_flotant_marsh(self) -> xr.DataArray:
        """
        Calculate percent of 480m cell that is flotant marsh.
        Fresh Marsh: 20

        Return:
            array of flotant marsh meeting both criteria, at 60m resolution.
        """
        da = xr.open_dataarray(self.flotant_marsh_path)
        da = da["band" == 0]
        # reproject to match hsi grid
        da = self._reproject_match_to_dem(da)

        # define which value is flotant marsh in raster key
        flotant_marsh = da == 4
        fresh_marsh = self.initial_veg_type["veg_type_subset"] == 20
        combined_flotant = flotant_marsh & fresh_marsh

        return combined_flotant

    def _get_water_depth_annual_mean(self) -> np.ndarray:
        """
        Calculates the difference between the avg annual WSE value and the DEM.

        Returns : np.ndarray
            Depth array
        """
        mean_wse = self.wse.mean(dim="time", skipna=True)["WSE_MEAN"]
        height = mean_wse - self.dem

        # upscale to 480m from 60m
        da_coarse = height.coarsen(y=8, x=8, boundary="pad").mean()
        return da_coarse.to_numpy()

    def _get_water_depth_monthly_mean_jan_aug(self) -> np.ndarray:
        """
        Calculates the difference between the mean WSE value for a
        selection of months and the DEM.

        Returns : np.ndarray
            Depth array
        """
        # Filter by month first
        jan_aug = [1, 2, 3, 4, 5, 6, 7, 8]
        # filter_jan_aug = self.wse.sel(dim="time".dt.month.isin(jan_aug))
        filter_jan_aug = self.wse.sel(time=self.wse["time"].dt.month.isin(jan_aug))

        # Calc mean
        mean_monthly_wse_jan_aug = filter_jan_aug.mean(dim="time", skipna=True)[
            "WSE_MEAN"
        ]
        height = mean_monthly_wse_jan_aug - self.dem

        # upscale to 480m from 60m
        da_coarse = height.coarsen(y=8, x=8, boundary="pad").mean()
        return da_coarse.to_numpy()

    def _get_water_depth_monthly_mean_sept_dec(self) -> np.ndarray:
        """
        Calculates the difference between the mean WSE value for a
        selection of months and the DEM.

        Returns : np.ndarray
            Depth array
        """

        # Filter by month first
        sept_dec = [9, 10, 11, 12]
        filter_sept_dec = self.wse.sel(time=self.wse["time"].dt.month.isin(sept_dec))

        # Calc mean
        mean_monthly_wse_sept_dec = filter_sept_dec.mean(dim="time", skipna=True)[
            "WSE_MEAN"
        ]
        height = mean_monthly_wse_sept_dec - self.dem

        # upscale to 480m from 60m
        da_coarse = height.coarsen(y=8, x=8, boundary="pad").mean()
        return da_coarse.to_numpy()

    def _create_output_dirs(self):
        """Create an output location for state variables, model config,
        input data, and QC plots.

        (No logging because logger needs output location for log file first.)
        """
        output_dir_name = f"HSI_{self.sim_start_time}"

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

    def _create_output_file(self, params: dict):
        """HSI: Create NetCDF file for data output.

        Parameters
        ----------
        params : dict
            Dict of filename attributes, specified in
            `utils.generate_filename()`

        Returns
        -------
        None
        """

        file_name = utils.generate_filename(
            params=params,
            base_path=self.timestep_output_dir,
            parameter="DATA",
        )

        self.netcdf_filepath = os.path.join(self.output_dir_path, f"{file_name}.nc")

        # load DEM, use coords
        da = xr.open_dataarray(self.dem_path)
        da = da["band" == 0]
        # coarsen to 480m grid cell
        da = da.coarsen(x=8, y=8, boundary="pad").mean()

        x = da.coords["x"].values
        y = da.coords["y"].values

        # Define the new time coordinate
        time_range = pd.date_range(
            start=f"{self.water_year_start}-10-01",
            end=f"{self.water_year_end}-10-01",
            freq="YS-OCT",
        )  # Annual start

        # Create an empty Dataset with only coordinates
        ds = xr.Dataset(coords={"time": time_range, "y": y, "x": x})
        ds = ds.rio.write_crs("EPSG:6344")

        # Add metadata
        # ds["veg_type"].attrs["description"] = "Vegetation type classification"
        # ds["veg_type"].attrs["units"] = "Category"
        # ds["maturity"].attrs["description"] = "Vegetation maturity in years"
        # ds["maturity"].attrs["units"] = "Years"

        # Save initial file
        ds.to_netcdf(self.netcdf_filepath, mode="w", format="NETCDF4")
        ds.close()
        da.close()
        self._logger.info("Initialized NetCDF file: %s", self.netcdf_filepath)

    def _append_hsi_vars_to_netcdf(self, timestep: pd.DatetimeTZDtype):
        """Append timestep data to the NetCDF file.

        Parameters
        ----------
        timestep : pd.DatetimeTZDtype
            Pandas datetime object of current timestep.

        Returns
        -------
        None
        """
        timestep_str = timestep.strftime("%Y-%m-%d")
        # Open existing NetCDF file
        ds = xr.open_dataset(self.netcdf_filepath)

        hsi_variables = {
            "alligator_hsi": self.alligator.hsi,
            "alligator_si_1": self.alligator.si_1,
            "alligator_si_2": self.alligator.si_2,
            "alligator_si_3": self.alligator.si_3,
            "alligator_si_4": self.alligator.si_4,
            "alligator_si_5": self.alligator.si_5,
            #
            "bald_eagle_hsi": self.baldeagle.hsi,
            "bald_eagle_si_1": self.baldeagle.si_1,
            "bald_eagle_si_2": self.baldeagle.si_2,
            "bald_eagle_si_3": self.baldeagle.si_3,
            "bald_eagle_si_4": self.baldeagle.si_4,
            "bald_eagle_si_5": self.baldeagle.si_5,
            "bald_eagle_si_6": self.baldeagle.si_6,
            #
            "crawfish_hsi": self.crawfish.hsi,
            "crawfish_si_1": self.crawfish.si_1,
            "crawfish_si_2": self.crawfish.si_2,
            "crawfish_si_3": self.crawfish.si_3,
            "crawfish_si_4": self.crawfish.si_4,
            # "black_bear_hsi": self.black_bear.hsi,
        }

        for var_name, data in hsi_variables.items():
            # create var if not already existing, with full timeseries of nan
            if var_name not in ds:
                ds[var_name] = (
                    ["time", "y", "x"],
                    np.full((len(ds.time), len(ds.y), len(ds.x)), np.nan),
                )

            # assign values for timestep
            # not dtype rules needed here as 480m cell is light
            ds[var_name].loc[{"time": timestep_str}] = data

        ds.close()
        ds.to_netcdf(self.netcdf_filepath, mode="a")
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
        self.timestep_output_dir = os.path.join(
            self.output_dir_path, f"{date.strftime('%Y%m%d')}"
        )
        self.timestep_output_dir_figs = os.path.join(
            self.timestep_output_dir,
            "figs",
        )
        os.makedirs(self.timestep_output_dir, exist_ok=True)
        os.makedirs(self.timestep_output_dir_figs, exist_ok=True)


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
