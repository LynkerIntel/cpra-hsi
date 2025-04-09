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
from rasterio.enums import Resampling
from skimage.morphology import disk, dilation

# import veg_logic
import hydro_logic
import plotting
import utils

import veg_transition as vt
from species_hsi import (
    alligator,
    crawfish,
    baldeagle,
    gizzardshad,
    bass,
    bluecrab,
    blackbear,
)


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

        with open(config_file, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        # fetch raster data paths
        self.dem_path = self.config["raster_data"].get("dem_path")
        self.wse_directory_path = self.config["raster_data"].get(
            "wse_directory_path"
        )
        self.wse_domain_path = self.config["raster_data"].get(
            "wse_domain_raster"
        )
        self.veg_base_path = self.config["raster_data"].get("veg_base_raster")
        self.veg_type_path = self.config["raster_data"].get("veg_type_path")
        self.veg_keys_path = self.config["raster_data"].get("veg_keys")
        self.salinity_path = self.config["raster_data"].get("salinity_raster")

        self.flotant_marsh_path = self.config["raster_data"].get(
            "flotant_marsh_raster"
        )
        # self.flotant_marsh_keys_path = self.config["raster_data"].get("flotant_marsh_keys")

        # TODO Add this path to config - Issue #40 on GitHub
        # self.bluecrab_lookup_table_path = self.config["???"].get("bluecrab_lookup_table")

        # simulation
        self.water_year_start = self.config["simulation"].get(
            "water_year_start"
        )
        self.water_year_end = self.config["simulation"].get("water_year_end")
        self.run_hsi = self.config["simulation"].get("run_hsi")
        self.analog_sequence = self.config["simulation"].get(
            "wse_sequence_input"
        )
        self.hydro_domain_flag = self.config["simulation"].get(
            "hydro_domain_flag"
        )
        self.blue_crab_lookup_path = self.config["simulation"].get(
            "blue_crab_lookup_table"
        )

        # metadata
        self.metadata = self.config["metadata"]

        # output
        self.output_base_dir = self.config["output"].get("output_base")

        # Pretty-print the configuration
        config_pretty = yaml.dump(
            self.config,
            default_flow_style=False,
            sort_keys=False,
        )

        # NetCDF data output
        sim_length = self.water_year_end - self.water_year_start

        self.file_params = {
            "model": self.metadata.get("model"),
            "scenario": self.metadata.get("scenario"),
            "group": self.metadata.get("group"),
            "wpu": "AB",
            "io_type": "O",
            "time_freq": "ANN",  # for annual output
            "year_range": f"01_{str(sim_length + 1).zfill(2)}",
            # "parameter": "NA",
        }

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
        self.dem_480 = self._load_dem(cell=True)
        self.veg_keys = self._load_veg_keys()
        self.edge = self._calculate_edge()
        self.initial_veg_type = self._load_veg_initial_raster(
            xarray=True,
            all_types=True,
        )
        self.flotant_marsh = self._calculate_flotant_marsh()
        self.hydro_domain = self._load_hecras_domain_raster()
        self.hydro_domain_480 = self._load_hecras_domain_raster(cell=True)

        # Get pct cover for prevously defined static variables
        self._calculate_pct_cover_static()

        # Dynamic Variables
        self.wse = None
        self.maturity = np.zeros_like(self.dem)
        self.water_depth_annual_mean = None
        self.veg_ts_out = None  # xarray output for timestep
        self.water_depth_monthly_mean_jan_aug = None
        self.water_depth_monthly_mean_sept_dec = None
        self.water_depth_monthly_mean_jan_aug_cm = None

        # HSI models
        self.alligator = None
        self.crawfish = None
        self.baldeagle = None
        self.gizzardshad = None
        self.bass = None
        self.blackbear = None
        self.bluecrab = None

        # datasets
        self.pct_cover_veg = None
        self._load_blue_crab_lookup()

        # HSI Variables
        self.pct_open_water = None
        # self.avg_water_depth_rlt_marsh_surface = None
        self.mean_annual_salinity = None
        self.mean_annual_temperature = None  # TODO: is never assigned a value

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
        # self.pct_marsh = None # not currently in use

        # gizzard shad vars
        self.tds_summer_growing_season = None  # ideal always
        self.avg_num_frost_free_days_growing_season = None  # ideal always
        self.mean_weekly_summer_temp = (
            None  # ideal (HEC-RAS?) SI3 = 25 degrees C
        )
        self.max_do_summer = None  # ideal HEC-RAS SI4 = 6ppm
        self.water_lvl_spawning_season = None  # ideal always
        self.mean_weekly_temp_reservoir_spawning_season = (
            None  # ideal HEC-RAS SI6 = 20 degrees
        )
        # only var to def for hec-ras 2.12.24  (separating (a)prt veg and (b)depth)
        self.pct_vegetated = None
        self.water_depth_spawning_season = None

        # black bear
        self.pct_soft_mast = None
        self.pct_hard_mast = None
        self.pct_no_mast = None

        self.num_soft_mast_species = None  # always ideal
        self.basal_area_hard_mast = None  # always ideal
        self.num_hard_mast_species = None  # always ideal

        self._create_output_file(self.file_params)

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
            run_metadata_dir = os.path.join(
                self.output_dir_path, "run-metadata"
            )
            os.makedirs(
                run_metadata_dir, exist_ok=True
            )  # Ensure directory exists
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

        Params
        -------
        date: pd.DatetimeTZDtype
            current datetime as pandas dt type
        """
        self.current_timestep = date
        wy = date.year

        self._logger.info("starting timestep: %s", date)
        self._create_timestep_dir(date)

        # water depth vars --------------------------------------
        self.wse = self.load_wse_wy(wy, variable_name="WSE_MEAN")
        self.wse = self._reproject_match_to_dem(self.wse)  # TEMPFIX
        self.water_depth_annual_mean = self._get_depth_filtered()
        self.water_depth_monthly_mean_jan_aug_cm = self._get_depth_filtered(
            months=[1, 2, 3, 4, 5, 6, 7, 8]
        )
        self.water_depth_monthly_mean_sept_dec = self._get_depth_filtered(
            months=[9, 10, 11, 12]
        )
        self.water_depth_spawning_season = self._get_depth_filtered(
            months=[4, 5, 6]
        )

        # load VegTransition output ----------------------------------
        self.veg_type = self._load_veg_type()

        # veg based vars ----------------------------------------------
        self._calculate_pct_cover()
        self.mean_annual_salinity = hydro_logic.habitat_based_salinity(
            self.veg_type,
            domain=self.hydro_domain,
            cell=True,
        )
        self._calculate_mast_percentage()
        self._calculate_near_forest(radius=4)

        # run ---------------------------------------------------------
        if self.run_hsi:

            self.alligator = alligator.AlligatorHSI.from_hsi(self)
            self.crawfish = crawfish.CrawfishHSI.from_hsi(self)
            self.baldeagle = baldeagle.BaldEagleHSI.from_hsi(self)
            self.gizzardshad = gizzardshad.GizzardShadHSI.from_hsi(self)
            self.bass = bass.BassHSI.from_hsi(self)
            self.bluecrab = bluecrab.BlueCrabHSI.from_hsi(self)
            self.blackbear = blackbear.BlackBearHSI.from_hsi(self)

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
        simulation_period = range(
            self.water_year_start, self.water_year_end + 1
        )
        self._logger.info(
            "Running model for: %s timesteps", len(simulation_period)
        )

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
        time_str = self.current_timestep.strftime("%Y%m%d")
        ds = xr.open_dataset(self.veg_type_path)
        da = ds.sel({"time": time_str})["veg_type"]
        ds.close()  # Ensure file closure
        return da

    def _calculate_pct_cover(self):
        """Get percent coverage for each 480m cell, based on 60m veg type pixels. This
        function is called for every HSI timestep.

        Derived from VegTransition Output

        VEG TYPES AND THIER MAPPED NUMBERS FROM
        2  Developed High Intensity
        3  Developed Medium Intensity
        4  Developed Low Intensity
        5  Developed Open Space
        6  Cultivated Crops
        7  Pasture/Hay
        8  Grassland/Herbaceous
        9  Upland - Mixed Deciduous Forest
        10  Upland - Mixed Evergreen Forest
        11  Upland Mixed Forest
        12  Upland Scrub/Shrub
        13  Unconsolidated Shore
        14  Bare Land
        15  Zone V (Upper BLH)
        16  Zone IV (Middle BLH)
        17  Zone III (Lower BLH)
        18  Zone II (Swamp)
        19  Fresh Shrubs
        20  Fresh Marsh
        21  Intermediate Marsh
        22  Brackish Marsh
        23  Saline Marsh
        24  Palustrine Aquatic Bed
        25  Estuarine Aquatic Bed
        26  Open Water
        """
        # note: masking for pct cover vars cannot occur here, because there
        # are limited pixels where the hydro domain exceeds the dem domain.
        # (these are masked out later)
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
            boundary="pad",
        )
        # ds_swamp_blh = ds_swamp_blh.where(~np.isnan(self.hydro_domain_480))

        ds_vegetated = utils.generate_pct_cover_custom(
            data_array=self.veg_type,
            veg_types=[
                v for v in range(15, 25)
            ],  # these are everything but open water
            x=8,
            y=8,
            boundary="pad",
        )
        # ds_vegetated = ds_vegetated.where(~np.isnan(self.hydro_domain_480))

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

        # Vegetated 15-25
        self.pct_vegetated = ds_vegetated.to_numpy()

        # Marsh 20-23
        # self.pct_marsh = ds_marsh.to_numpy() # not currently in use

        # Zone V, IV, III, (BLH's) II (swamp)
        self.pct_swamp_bottom_hardwood = ds_swamp_blh.to_numpy()

    def _calculate_pct_cover_static(self):
        """Get percent coverage variables for each 480m cell, based on 60m veg type pixels.
        This method is called during initialization, for static variables that
        vary spatially but not temporally.
        """
        self.pct_dev_upland = utils.generate_pct_cover_custom(
            data_array=self.initial_veg_type,
            # these are the dev'd (4) and upland (4)
            veg_types=[2, 3, 4, 5, 9, 10, 11, 12],
            x=8,
            y=8,
            boundary="pad",
        ).to_numpy()

        # Flotant marsh for baldeagle
        self.pct_flotant_marsh = utils.coarsen_and_reduce(
            da=self.flotant_marsh,
            veg_type=True,
            x=8,
            y=8,
            boundary="pad",
        ).to_numpy()

    @staticmethod
    def _calculate_near_landtype(
        landcover_arr: np.ndarray, landtype_true: list, radius: int
    ) -> np.ndarray:
        """Generalized method to get percentage of 480m cell with within
        radius of a land cover type or list of land cover types. Uses
        "initial vegtype" array, i.e. the initial conditions, with non-veg
        land cover classes included.

        Parameters
        -----------
        landtype : list
            list of land type int values considered "True", i.e. non-vegetated land

        radius : int
            radius to use for the circular kernel, to determine nearby land cover

        Returns
        --------
        istype_array : np.ndarray
            Array of pct cover of pixels meeting the defined criteria
        """
        istype_bool = np.isin(landcover_arr, landtype_true)
        nottype_bool = ~istype_bool

        disk_kernel = disk(radius)  # circular grid w/ radius (kernel)
        istype_expanded = dilation(istype_bool, disk_kernel)

        # mask to keep only non-type near type
        near_types = istype_expanded & nottype_bool

        # get percent (% of 480m cell, from 60m pixels)
        near_types_da = xr.DataArray(near_types.astype(float))
        pct_near = (
            near_types_da.coarsen(x=8, y=8, boundary="pad").mean() * 100
        ).to_numpy()

        return pct_near

    def _calculate_pct_area_influence(self):
        """Percent of evaluation area inside of zones of influence defined by
        radii 5.7 km around towns; 3.5 km around cropland; and 1.1 km around
        residences.

        Radiuses are defined by circular (disk) kernel, which expands True
        pixels outward by r.

        5,700 / 60 = 95 pixels
        3,500 / 60 = 58.33 pixels

        """
        towns = [2, 3, 4, 5]
        croplands = [6, 7, 8]

        # ---------- towns --------------
        self.pct_near_towns = self._calculate_near_landtype(
            landcover_arr=self.veg_type,
            landtype_true=towns,
            radius=95,
        )

        # ---------- croplands ------------
        self.pct_near_croplands = self._calculate_near_landtype(
            landcover_arr=self.veg_type,
            landtype_true=croplands,
            radius=58,
        )

        self.pct_near_towns_croplands = (
            self.pct_near_croplands & self.pct_near_towns
        )

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
            [(neighbor == open_water) for neighbor in neighbors],
            dim="direction",
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

    def _get_depth_filtered(
        self, months: None | list[int] = None
    ) -> np.ndarray:
        """Calls the VegTransition _get_depth(), then adds a time
        filter (if supplied) and then resample to 480m cell size.

        Parameters
        ----------
        months : list (optional)
            List of months to average water depth over. If a list is not
            provided, the default is all months

        Return
        ------
        da_coarse : xr.DataArray
            A water depth data, averaged over a list of months (if provided)
            and then downscaled to 480m.
        """
        ds = super()._get_depth()  # VegTransition._get_depth()

        if not months:
            months = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

        filtered_ds = ds.sel(time=self.wse["time"].dt.month.isin(months))
        ds = filtered_ds.mean(dim="time", skipna=True)["WSE_MEAN"]

        da_coarse = ds.coarsen(y=8, x=8, boundary="pad").mean()
        return da_coarse.to_numpy()

    def _calculate_mast_percentage(self):
        """Calculate percetange of canopy cover for mast classifications, as a
        summation of `%_cover * %_mast` for zones II through V. The pct_cover
        arrays are 480m, so the output is also 480m.


        | Zone | Growth (cm/yr) | Hard Mast (%) | Soft Mast (%) | No Mast (%) |
        |------|----------------|---------------|---------------|-------------|
        | II   | 0.52           | 0.00          | 0.53          | 0.47        |
        | III  | 0.63           | 0.39          | 0.61          | 0.00        |
        | IV   | 0.66           | 0.28          | 0.42          | 0.30        |
        | V    | 0.68           | 1.00          | 0.00          | 0.00        |

        """
        soft_mast = {"II": 0.53, "III": 0.61, "IV": 0.42, "V": 0.00}
        hard_mast = {"II": 0.0, "III": 0.39, "IV": 0.28, "V": 1.00}
        no_mast = {"II": 0.47, "III": 0.0, "IV": 0.3, "V": 0.0}

        # calculate as a weighted sum of zones
        self.pct_soft_mast = (
            self.pct_zone_ii * soft_mast["II"]
            + self.pct_zone_iii * soft_mast["III"]
            + self.pct_zone_iv * soft_mast["IV"]
            + self.pct_zone_v * soft_mast["V"]
        ) * 100

        self.pct_hard_mast = (
            self.pct_zone_ii * hard_mast["II"]
            + self.pct_zone_iii * hard_mast["III"]
            + self.pct_zone_iv * hard_mast["IV"]
            + self.pct_zone_v * hard_mast["V"]
        ) * 100

        self.pct_no_mast = (
            self.pct_zone_ii * no_mast["II"]
            + self.pct_zone_iii * no_mast["III"]
            + self.pct_zone_iv * no_mast["IV"]
            + self.pct_zone_v * no_mast["V"]
        ) * 100

    def _calculate_near_forest(self, radius: int = 4) -> np.ndarray:
        """Percent of area in nonforested cover types ≤ 250m from forested cover types.
        Computes non-forested pixels within a disk-shaped neighborhood of forest pixels,
        with radius=4 (i.e. 240m).

        TODO: confirm forest pixels

        parameters
        ----------
        radius : int
            radius of the neighborhood in pixels
        """
        forest_types = [15, 16, 17, 18]

        forested_mask = np.isin(self.veg_type, forest_types)
        non_forested_mask = ~forested_mask  # invert mask, i.e. everything else

        disk_kernel = disk(radius)  # circular grid w/ radius (kernel)
        expanded_forest = dilation(forested_mask, disk_kernel)
        # mask to keep only non-forested near forest
        near_forest_mask = expanded_forest & non_forested_mask

        near_forest_mask_da = xr.DataArray(near_forest_mask.astype(float))

        self.pct_near_forest = (
            near_forest_mask_da.coarsen(
                dim_0=8, dim_1=8, boundary="pad"
            ).mean()  # uses default coord names (dim_0, 1)
        ) * 100  # UNIT: index to percent

    def _load_blue_crab_lookup(self):
        """
        Read blue crab lookup table
        """
        self.blue_crab_lookup_table = pd.read_csv(self.blue_crab_lookup_path)

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

        self.netcdf_filepath = os.path.join(
            self.output_dir_path, f"{file_name}.nc"
        )

        # Load DEM as a template for coordinates
        da = xr.open_dataarray(self.dem_path)
        da = da.squeeze(drop="band")  # Drop 'band' dimension if it exists
        da = da.rio.write_crs("EPSG:6344")  # Assign CRS

        # Resample to 480m resolution, using rioxarray, with preserved correct coords and
        # assigns GeoTransform to spatial_ref. xr.coarsen() does not produce correct
        # projected coords.
        da = da.rio.reproject(
            da.rio.crs, resolution=480, resampling=Resampling.average
        )

        # use new 480m coords
        x = da.coords["x"].values
        y = da.coords["y"].values

        # Define the new time coordinate
        time_range = pd.date_range(
            start=f"{self.water_year_start}-10-01",
            end=f"{self.water_year_end}-10-01",
            freq="YS-OCT",  # Annual start
        )

        ds = xr.Dataset(
            # initialize w/ no data vars
            # {
            #     "initial_conditions": (
            #         ["time", "y", "x"],
            #         data_values,
            #         {"grid_mapping": "crs"},
            #     ),  # Link CRS variable
            # },
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
                "time": ("time", time_range, {"long_name": "Time"}),
            },
            attrs={"title": "HSI"},
        )

        ds = ds.rio.write_crs("EPSG:6344")

        # Save dataset to NetCDF
        ds.to_netcdf(self.netcdf_filepath)
        self._logger.info(
            "Initialized NetCDF file with CRS: %s", self.netcdf_filepath
        )

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

        hsi_variables = {
            # alligator
            "alligator_hsi": (self.alligator.hsi, np.float32),
            "alligator_si_1": (self.alligator.si_1, np.float32),
            "alligator_si_2": (self.alligator.si_2, np.float32),
            "alligator_si_3": (self.alligator.si_3, np.float32),
            "alligator_si_4": (self.alligator.si_4, np.float32),
            "alligator_si_5": (self.alligator.si_5, np.float32),
            # bald eagle
            "bald_eagle_hsi": (self.baldeagle.hsi, np.float32),
            "bald_eagle_si_1": (self.baldeagle.si_1, np.float32),
            "bald_eagle_si_2": (self.baldeagle.si_2, np.float32),
            "bald_eagle_si_3": (self.baldeagle.si_3, np.float32),
            "bald_eagle_si_4": (self.baldeagle.si_4, np.float32),
            "bald_eagle_si_5": (self.baldeagle.si_5, np.float32),
            "bald_eagle_si_6": (self.baldeagle.si_6, np.float32),
            # crawfish
            "crawfish_hsi": (self.crawfish.hsi, np.float32),
            "crawfish_si_1": (self.crawfish.si_1, np.float32),
            "crawfish_si_2": (self.crawfish.si_2, np.float32),
            "crawfish_si_3": (self.crawfish.si_3, np.float32),
            "crawfish_si_4": (self.crawfish.si_4, np.float32),
            # gizzard shad
            "gizzard_shad_hsi": (self.gizzardshad.hsi, np.float32),
            "gizzard_shad_si_1": (self.gizzardshad.si_1, np.float32),
            "gizzard_shad_si_2": (self.gizzardshad.si_2, np.float32),
            "gizzard_shad_si_3": (self.gizzardshad.si_3, np.float32),
            "gizzard_shad_si_4": (self.gizzardshad.si_4, np.float32),
            "gizzard_shad_si_5": (self.gizzardshad.si_5, np.float32),
            "gizzard_shad_si_6": (self.gizzardshad.si_6, np.float32),
            "gizzard_shad_si_7": (self.gizzardshad.si_7, np.float32),
            # bass
            "bass_hsi": (self.bass.hsi, np.float32),
            "bass_si_1": (self.bass.si_1, np.float32),
            "bass_si_2": (self.bass.si_2, np.float32),
            # species input vars
            "water_depth_annual_mean": (
                self.water_depth_annual_mean,
                np.float32,
            ),
            "water_depth_monthly_mean_jan_aug": (
                self.water_depth_monthly_mean_jan_aug,
                np.float32,
            ),
            "water_depth_monthly_mean_sept_dec": (
                self.water_depth_monthly_mean_sept_dec,
                np.float32,
            ),
            "water_depth_spawning_season": (
                self.water_depth_spawning_season,
                np.float32,
            ),
            "pct_open_water": (self.pct_open_water, np.float32),
            "mean_annual_salinity": (self.mean_annual_salinity, np.float32),
            "mean_annual_temperature": (
                self.mean_annual_temperature,
                np.float32,
            ),
            "pct_swamp_bottom_hardwood": (
                self.pct_swamp_bottom_hardwood,
                np.float32,
            ),
            "pct_fresh_marsh": (self.pct_fresh_marsh, np.float32),
            "pct_intermediate_marsh": (
                self.pct_intermediate_marsh,
                np.float32,
            ),
            "pct_brackish_marsh": (self.pct_brackish_marsh, np.float32),
            "pct_saline_marsh": (self.pct_saline_marsh, np.float32),
            "pct_zone_v": (self.pct_zone_v, np.float32),
            "pct_zone_iv": (self.pct_zone_iv, np.float32),
            "pct_zone_iii": (self.pct_zone_iii, np.float32),
            "pct_zone_ii": (self.pct_zone_ii, np.float32),
            "pct_fresh_shrubs": (self.pct_fresh_shrubs, np.float32),
            "pct_bare_ground": (self.pct_bare_ground, np.float32),
            "pct_dev_upland": (self.pct_dev_upland, np.float32),
            "pct_flotant_marsh": (self.pct_flotant_marsh, np.float32),
            "pct_vegetated": (self.pct_vegetated, np.float32),
        }

        # Open existing NetCDF file
        with xr.open_dataset(self.netcdf_filepath) as ds:

            for var_name, (data, dtype) in hsi_variables.items():
                if data is None:
                    self._logger.warning(
                        "Array '%s' is None, skipping save.", var_name
                    )

                else:
                    # Check if the variable exists in the dataset, if not, initialize it
                    if var_name not in ds:
                        shape = (len(ds.time), len(ds.y), len(ds.x))
                        default_value = False if dtype == bool else np.nan
                        ds[var_name] = (
                            ["time", "y", "x"],
                            np.full(shape, default_value, dtype=dtype),
                            {"grid_mapping": "crs"},
                        )

                    # Handle 'condition' variables (booleans)
                    if dtype == bool:
                        data = np.nan_to_num(data, nan=False).astype(bool)

                    # Assign the data to the dataset for the specific time step
                    ds[var_name].loc[{"time": timestep_str}] = data.astype(
                        ds[var_name].dtype
                    )

        ds.close()
        ds.to_netcdf(
            self.netcdf_filepath,
            mode="a",
        )
        # ds.to_netcdf(self.netcdf_filepath, mode="a")  # netcdf4 backend not preserving crs
        self._logger.info("Appended timestep %s to NetCDF file.", timestep_str)

    def post_process(self):
        """HSI post process

        Opens file and then crops to hydro domain
        """
        with xr.open_dataset(self.netcdf_filepath) as ds:
            ds_out = ds.where(~np.isnan(self.hydro_domain_480)).copy(deep=True)

            # hack: remove the file before writing to prevent conflicts
            # not sure why the file would be open at this point
            if os.path.exists(self.netcdf_filepath):
                os.remove(self.netcdf_filepath)

            ds_out.close()
            ds_out.to_netcdf(self.netcdf_filepath, mode="w")

    # def _create_timestep_dir(self, date: pd.DatetimeTZDtype):
    #     """Create output directory for the current timestamp, where
    #     figures and output rasters will be saved.

    #     Parameters
    #     ----------
    #     timestep : pd.DatetimeTZDtype
    #         Pandas datetime object of current timestep.

    #     Returns
    #     -------
    #     None
    #     """
    #     self.timestep_output_dir = os.path.join(
    #         self.output_dir_path, f"{date.strftime('%Y%m%d')}"
    #     )
    #     self.timestep_output_dir_figs = os.path.join(
    #         self.timestep_output_dir,
    #         "figs",
    #     )
    #     os.makedirs(self.timestep_output_dir, exist_ok=True)
    #     os.makedirs(self.timestep_output_dir_figs, exist_ok=True)


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
