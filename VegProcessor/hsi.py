import logging
import yaml
import xarray as xr
import numpy as np
import pandas as pd
import pprint


import os
from typing import Optional
import rioxarray  # used for tif output
from datetime import datetime
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from scipy.ndimage import label

# import veg_logic
import hydro_logic
import utils
import validate

import veg_transition as vt
from output_vars import get_hsi_variables

from species_hsi import (
    alligator,
    catfish,
    crawfish,
    baldeagle,
    gizzardshad,
    bass,
    bluecrab,
    blackbear,
    blhwva,
    swampwva,
    blackcrappie,
)


class HSI(vt.VegTransition):
    """HSI model framework.

    This class handles the creation and execution of HSI models, using
    methods from the inherited `VegTransition` parent class to provide
    the execution framework, and HSI variables defined below. HSI variables
    can be static (i.e. calculated before model steps forward and used for
    all timestep) or they can be dynamic (updated for each timestep).

    The HSI framework is designed to run with highly varying numbers of
    input variables. All variables are initialized as "None" and either
    replaced with actual data, or replaced with stand-in values during
    execution. The stand-in values are either "1" indicating an ideal
    SI (suitability index score), or an empirical value provided to the
    S.I. function that results in a approximate scoring. All other vars
    are created from one of a range of hydrologic model outputs.

    In general, the code structure favors creating HSI input variables as
    numpy arrays, and adding them as `HSI` class attributes, which are
    updated each timestep. This balances the tradeoff of memory use
    (many arrays are created at each step) with observability; it is easier
    to debug issues when all arrays can be accessed after a timestep. Future
    revision to the codebase may want to favor more memory-efficient methods,
    once the logic has been tested and verified.

    All *HSI variable* arrays in `HSI` are assumed to be at 480m (a.k.a "cell"
    resolution) unless otherwise noted, typically with a "_60m" suffix
    (a.k.a "pixel" resolution).
    """

    def __init__(self, config_file: str, log_level: int = logging.INFO):
        """
        Initialize by setting up logger, loading resource paths, and creating empty
        arrays for state variables.

        Parameters:
        -----------
        config_file : str
            Path to configuration YAML
        log_level : int
            Level of vebosity for logging. Hardcoded for now.
        """
        self.sim_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config_path = config_file

        with open(config_file, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        validate.validate_config(self.config, config_file)
        self._load_config_attributes()

        # Generate filename early so it's available for logger and metadata files
        self.file_name = utils.generate_filename(
            params=self.file_params,
            hydro_source_model=self.metadata.get("hydro_source_model"),
        )

        self._create_output_dirs()
        self.current_timestep = None  # set in step() method
        self._setup_logger(log_level)
        self.timestep_output_dir = None  # set in step() method

        config_pretty = yaml.dump(
            self.config,
            default_flow_style=False,
            sort_keys=False,
        )
        self.sequence_mapping = utils.load_sequence_csvs("./sequences/")
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
        self.human_influence = None
        self.hydro_domain = self._load_hydro_domain_raster(as_float=True)
        self.hydro_domain_480 = self._load_hydro_domain_raster(cell=True)

        # Dynamic Variables --------------------------------------------
        self.maturity = None  # 60m, used by HSI
        self.maturity_480 = None  # 480m, passed directly to `blhwva.py`
        self.water_depth_annual_mean = None
        self.veg_ts_out = None  # xarray output for timestep
        self.water_depth_jan_aug_mean = None
        self.water_depth_oct_dec_mean = None
        self.water_depth_july_august_mean = None

        self.water_depth_july_august_mean_60m = None
        self.water_depth_july_sept_mean_60m = None
        self.water_depth_may_july_mean_60m = None

        # HSI models
        self.alligator = None
        self.baldeagle = None
        self.bass = None
        self.blackbear = None
        self.blackcrappie = None
        self.blhwva = None
        self.bluecrab = None
        self.catfish = None
        self.crawfish = None
        self.gizzardshad = None
        self.swampwva = None

        # datasets
        self.pct_cover_veg = None
        self._load_blue_crab_lookup()

        # HSI Variables
        self.pct_open_water = None
        self.water_temperature = None  # the source xr.dataset
        self.water_temperature_annual_mean = None

        # 60m water temperature for pools and backwaters
        self.water_temperature_feb_march_mean_60m = None
        self.water_temperature_may_july_mean_60m = None
        self.water_temperature_july_sept_mean_60m = None
        self.water_temperature_july_august_mean_60m = None

        self.salinity = None  # the source xr.dataset
        self.salinity_annual_mean = None
        self.salinity_max_april_sept = None
        self.salinity_max_july_sept = None
        self.salinity_max_may_july = None

        self.velocity = None

        self.pct_swamp_bottom_hardwood = None
        self.pct_fresh_marsh = None
        self.pct_intermediate_marsh = None
        self.pct_brackish_marsh = None
        self.pct_saline_marsh = None
        self.pct_shrub_scrub = None
        self.pct_shrub_scrub_midstory = None

        self.pct_zone_v = None
        self.pct_zone_iv = None
        self.pct_zone_iii = None
        self.pct_zone_ii = None
        self.pct_fresh_shrubs = None
        self.pct_blh = None

        self.pct_bare_ground = None
        self.pct_pools_july_sept_mean = None
        self.pct_pools_april_sept_mean = None

        # gizzard shad vars
        self.tds_summer_growing_season = None  # ideal always
        self.avg_num_frost_free_days_growing_season = None  # ideal always
        self.mean_weekly_summer_temp = (
            None  # ideal always (HEC-RAS?) SI3 = 25 degrees C
        )
        self.max_do_summer = None  # ideal HEC-RAS SI4 = 6ppm
        self.water_lvl_spawning_season = None  # ideal always
        self.water_lvl_change = None  # ideal
        self.is_veg_inundated = None  # ideal
        # ideal HEC-RAS SI6 = 20 degrees
        self.mean_weekly_temp_reservoir_spawning_season = None

        # only var to def for hec-ras 2.12.24  (separating (a)prt veg and (b)depth)
        self.pct_vegetated = None
        self.water_depth_april_june_mean = None

        # tree mast
        self.pct_soft_mast = None
        self.pct_hard_mast = None
        self.pct_no_mast = None
        self.pct_has_mast = None
        self.story_class = np.zeros_like(self.dem)

        self.num_soft_mast_species = None  # always ideal
        self.basal_area_hard_mast = None  # always ideal
        self.num_hard_mast_species = None  # always ideal
        self.pct_near_forest = None

        self.forested_connectivity_cat = None
        self.pct_overstory = None
        self.pct_midstory = None
        self.pct_understory = None
        self.maturity_dbh = None  # always ideal
        self.flood_duration = None  # TODO
        self.flow_exchange = None  # TODO
        self.salinity_mean_high_march_nov = None
        self.suit_trav_surr_lu = None  # always ideal
        self.disturbance = None  # always ideal

        # surrounding land use
        self.pct_forested_half_mi = None
        self.pct_abandoned_ag_half_mi = None
        self.pct_pasture_half_mi = None
        self.pct_active_ag_water_half_mi = None
        self.pct_nonhabitat_half_mi = None

        # black-crappie
        self.blackcrappie_max_monthly_avg_summer_turbidity = None
        self.blackcrappie_pct_cover_in_midsummer_pools_overflow_bw = (
            None  # set to ideal
        )
        self.blackcrappie_stream_gradient = None  # set to ideal
        self.blackcrappie_avg_vel_summer_flow_pools_bw = None
        self.blackcrappie_ph_year = None  # set to ideal
        # self.blackcrappie_most_suit_temp_in_midsummer_pools_bw_adult = None
        # self.blackcrappie_most_suit_temp_in_midsummer_pools_bw_juvenile = None
        # self.blackcrappie_avg_midsummer_temp_in_pools_bw_fry = None
        # self.blackcrappie_avg_spawning_temp_in_bw_embryo = None
        self.blackcrappie_min_do_in_midsummer_temp_strata = None
        self.blackcrappie_min_do_in_spawning_bw = None
        # self.blackcrappie_max_salinity_gs = None

        # catfish
        self.catfish_pct_cover_in_summer_pools_bw = None
        self.catfish_fpp_substrate_avg_summer_flow = None
        # self.catfish_avg_temp_in_midsummer_pools_bw = None
        self.catfish_grow_season_length_frost_free_days = None
        self.catfish_max_monthly_avg_summer_turbidity = None
        self.catfish_avg_min_do_in_midsummer_pools_bw = None
        # self.catfish_max_summer_salinity = None
        # self.catfish_avg_temp_in_spawning_embryo_pools_bw = None
        # self.catfish_max_salinity_spawning_embryo = None
        # self.catfish_avg_midsummer_temp_in_pools_bw_fry = None
        # self.catfish_max_summer_salinity_fry_juvenile = None
        # self.catfish_avg_midsummer_temp_in_pools_bw_juvenile = None
        self.catfish_avg_vel_summer_flow = None

        self._create_output_file(resolution=480)
        self._create_output_file(resolution=60)

    def _load_config_attributes(self):
        """Load configuration attributes from the config dictionary."""
        # fetch raster data paths
        self.dem_path = self.config["raster_data"].get("dem_path")
        self.wse_directory_path = self.config["raster_data"].get(
            "wse_directory_path"
        )
        self.hydro_domain_path = self.config["raster_data"].get(
            "hydro_domain_raster"
        )
        self.veg_base_path = self.config["raster_data"].get("veg_base_raster")
        self.veg_type_path = self.config["raster_data"].get("veg_type_path")
        self.veg_keys_path = self.config["raster_data"].get("veg_keys")
        self.salinity_path = self.config["raster_data"].get("salinity_raster")

        self.flotant_marsh_path = self.config["raster_data"].get(
            "flotant_marsh_raster"
        )
        # self.flotant_marsh_keys_path = self.config["raster_data"].get("flotant_marsh_keys")

        # simulation
        self.water_year_start = self.config["simulation"].get(
            "water_year_start"
        )
        self.water_year_end = self.config["simulation"].get("water_year_end")
        self.run_hsi = self.config["simulation"].get("run_hsi")
        self.analog_sequence = self.config["simulation"].get(
            "wse_sequence_input"
        )
        self.netcdf_hydro_path = self.config["raster_data"].get(
            "netcdf_hydro_path"
        )
        self.netcdf_salinity_path = self.config["raster_data"].get(
            "netcdf_salinity_path"
        )
        self.netcdf_water_temperature_path = self.config["raster_data"].get(
            "netcdf_water_temperature_path"
        )
        self.netcdf_velocity_path = self.config["raster_data"].get(
            "netcdf_velociy_path"
        )
        self.blue_crab_lookup_path = self.config["simulation"].get(
            "blue_crab_lookup_table"
        )
        self.years_mapping = self.config["simulation"].get("years_mapping")
        self.testing_radius = self.config["simulation"].get("testing_radius")
        self.hsi_run_species = self.config["simulation"].get(
            "hsi_run_species", []
        )

        # metadata
        self.metadata = self.config["metadata"]
        self.scenario_type = self.config["metadata"].get(
            "scenario", ""
        )  # empty str if missing

        # output
        self.output_base_dir = self.config["output"].get("output_base")

        # NetCDF data output
        sim_length = self.water_year_end - self.water_year_start + 1

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
                f"01_{str(sim_length).zfill(2)}"
            ),  # HSI does not include initial conditions (unlike VEG)
            "output_version": self.metadata.get("output_version"),
        }

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
        self.wy = date.year

        self._logger.info("starting timestep: %s", date)
        self._create_timestep_dir(date)

        # water depth vars -----------------------------------------------
        self.water_depth = self._load_depth_general(self.wy)
        self.water_depth_annual_mean = self._get_daily_depth_filtered()
        self.water_depth_jan_aug_mean = self._get_daily_depth_filtered(
            months=[1, 2, 3, 4, 5, 6, 7, 8],
        )
        self.water_depth_oct_dec_mean = self._get_daily_depth_filtered(
            months=[10, 11, 12],
        )
        self.water_depth_april_june_mean = self._get_daily_depth_filtered(
            months=[4, 5, 6],
        )
        self.water_depth_july_august_mean = self._get_daily_depth_filtered(
            months=[7, 8],
        )
        # 60m water depth for pools and backwaters logic -------------------
        self.water_depth_july_august_mean_60m = self._get_daily_depth_filtered(
            months=[7, 8],
            cell=False,
        )
        self.water_depth_july_sept_mean_60m = self._get_daily_depth_filtered(
            months=[7, 8, 9],
            cell=False,
        )
        self.water_depth_may_july_mean_60m = self._get_daily_depth_filtered(
            months=[5, 6, 7],
            cell=False,
        )

        # temperature vars -------------------------------------------
        self.water_temperature = self._load_water_temp_general(self.wy)

        if self.water_temperature is not None:
            self.water_temperature_annual_mean = (
                self._get_water_temperature_subset()
            )
            # 60m water temperature ----------------------------------------
            self.water_temperature_feb_march_mean_60m = (
                self._get_water_temperature_subset(months=[2, 3], cell=False)
            )
            self.water_temperature_may_july_mean_60m = (
                self._get_water_temperature_subset(
                    months=[5, 6, 7], cell=False
                )
            )
            self.water_temperature_july_august_mean_60m = (
                self._get_water_temperature_subset(months=[7, 8], cell=False)
            )
            self.water_temperature_july_sept_mean_60m = (
                self._get_water_temperature_subset(
                    months=[7, 8, 9], cell=False
                )
            )

        # load VegTransition output ----------------------------------
        self.veg_type = self._load_veg_type()
        self.maturity = self._load_maturity()
        self.maturity_480 = self._load_maturity(resample_cell=True)
        self.velocity = self._load_velocity_general(self.wy)

        # salinity vars -------------------------------------------------
        self.salinity = self._load_salinity_general(self.wy, cell=True)
        # only subset for Dataset() salinity (i.e. modeled)
        if isinstance(self.salinity, xr.Dataset):
            self.salinity_annual_mean = self._get_salinity_subset()
            self.salinity_max_april_sept = self._get_salinity_subset(
                months=[4, 5, 6, 7, 8, 9],
                method="max",
            )
            self.salinity_max_july_sept = self._get_salinity_subset(
                months=[7, 8, 9],
                method="max",
            )
            self.salinity_max_may_july = self._get_salinity_subset(
                months=[5, 6, 7],
                method="max",
            )
            self.salinity_mean_high_march_nov = self._get_salinity_subset(
                method="upper-pctile-mean",
                months=[3, 4, 5, 6, 7, 8, 9, 10, 11],
            )
        else:
            self.salinity_annual_mean = self.salinity

        # pct pools --------------------------------------------------
        self.pct_pools_july_sept_mean = self._get_pct_pools(
            months=[7, 8, 9],
            low=3,
            high=6,
        )
        self.pct_pools_april_sept_mean = self._get_pct_pools(
            months=[4, 5, 6, 7, 8, 9],
            low=0.5,
            high=3,
        )

        # veg based vars ----------------------------------------------
        self._calculate_pct_cover()
        self._calculate_mast_percentage()
        self._calculate_near_forest(radius=4)
        self._calculate_story_assignment()
        self._calculate_connectivity()
        self._calculate_shrub_scrub_midstory()

        # run ---------------------------------------------------------
        if self.run_hsi:
            species_map = {
                "alligator": alligator.AlligatorHSI,
                "catfish": catfish.RiverineCatfishHSI,
                "crawfish": crawfish.CrawfishHSI,
                "baldeagle": baldeagle.BaldEagleHSI,
                "gizzardshad": gizzardshad.GizzardShadHSI,
                "bass": bass.BassHSI,
                "bluecrab": bluecrab.BlueCrabHSI,
                "blackbear": blackbear.BlackBearHSI,
                "blhwva": blhwva.BottomlandHardwoodHSI,
                "swampwva": swampwva.SwampHSI,
                "blackcrappie": blackcrappie.BlackCrappieHSI,
            }

            # Run only species listed in config
            for species in self.hsi_run_species:
                if species in species_map:
                    setattr(self, species, species_map[species].from_hsi(self))

            self._append_480m_hsi_vars_to_netcdf(
                timestep=self.current_timestep
            )
            self._append_60m_hsi_vars_to_netcdf(timestep=self.current_timestep)

        self._logger.info("completed timestep: %s", date)
        self.current_timestep = None

    def run(self):
        """
        Run the vegetation transition model, with parameters defined in the configuration file.
        Start and end parameters are year, and handled as ints. No other frequency currently possible.
        """
        # run model forwards
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
        # static vars calcs outside of simulation loop
        self._calculate_static_vars()
        # run the model
        for wy in simulation_period:
            self.step(pd.to_datetime(f"{wy}-10-01"))
        # log data vars being supplied to the model
        # (after main loop to include all vars)
        self.log_data_attribute_types()
        self._logger.info("Simulation complete")
        logging.shutdown()

    def _load_veg_type(self) -> xr.DataArray:
        """Load VegTransition output.

        Returns : xr.DataArray
            a single WY, defined by `self.current_timestep`.
        """
        logging.info("Loading vegetation data.")
        time_str = self.current_timestep.strftime("%Y%m%d")
        ds = xr.open_dataset(self.veg_type_path)
        da = ds.sel({"time": time_str})["veg_type"]
        ds.close()  # Ensure file closure
        return da

    def _load_maturity(self, resample_cell: bool = False) -> np.ndarray:
        """Load forest maturity VegTransition output.

        Params
        -------
        resample : bool
            True if maturity should be resampled to 480m cell
            size using mean.

        Returns : xr.DataArray
            a single WY, defined by `self.current_timestep`.
        """
        logging.info("Loading maturity.")
        time_str = self.current_timestep.strftime("%Y%m%d")
        ds = xr.open_dataset(self.veg_type_path)
        da = ds.sel({"time": time_str})["maturity"]
        ds.close()  # Ensure file closure

        if resample_cell:
            da = da.coarsen(y=8, x=8, boundary="pad").mean()

        return da.to_numpy()

    def _load_water_temp_general(self, water_year: int) -> xr.Dataset | None:
        """Load water temperature data from either Delft3D or MIKE 21 models."""
        if self.netcdf_water_temperature_path is not None:
            self._logger.info(
                f"Loading water temperature data with universal daily method."
            )
            nc_path, analog_year = self._get_hydro_netcdf_path(
                water_year, hydro_variable="WTEMP"
            )
            self._logger.info("Loading file: %s", nc_path)

            ds = xr.open_dataset(
                nc_path,
                engine="h5netcdf",
                chunks="auto",
            )

            ds = utils.analog_years_handler(analog_year, water_year, ds)

            # # model specific var names: -----------------------------------------------
            # if self.file_params["hydro_source_model"] == "D3D":
            #     ds = ds.rename({"waterlevel": "height"})
            # if self.file_params["hydro_source_model"] == "MIK":
            #     ds = ds.rename({"water_level": "height"})
            # # extract height var as da
            # height_da = ds["sali"]

            # handle varied CRS metadata locations between model files-----------------
            try:
                # D3D & MIKE: CRS from crs variable's crs_wkt attribute
                crs_wkt = ds["crs"].attrs.get("crs_wkt")
                ds = ds.rio.write_crs(crs_wkt)

            except Exception as exc:
                raise ValueError(
                    "Unable to parse CRS from hydrologic input"
                ) from exc

            ds = self._reproject_match_to_dem(ds)
            return ds

        else:
            self._logger.info("Water Temperature not provided.")
            return None

    def _load_velocity_general(self, water_year: int) -> np.ndarray | None:
        """Load velocity data from either Delft3D or MIKE 21 models."""
        if self.netcdf_velocity_path is not None:
            self._logger.info(
                f"Loading velocity data with universal daily method."
            )
            nc_path, analog_year = self._get_hydro_netcdf_path(
                water_year, hydro_variable="VELOCITY"
            )
            self._logger.info("Loading file: %s", nc_path)

            ds = xr.open_dataset(
                nc_path,
                engine="h5netcdf",
                chunks="auto",
            )

            ds = utils.analog_years_handler(analog_year, water_year, ds)

            # # model specific var names: -----------------------------------------------
            # if self.file_params["hydro_source_model"] == "D3D":
            #     ds = ds.rename({"waterlevel": "height"})
            # if self.file_params["hydro_source_model"] == "MIK":
            #     ds = ds.rename({"water_level": "height"})
            # # extract height var as da
            # height_da = ds["sali"]

            # handle varied CRS metadata locations between model files-----------------
            try:
                # D3D & MIKE: CRS from crs variable's crs_wkt attribute
                crs_wkt = ds["crs"].attrs.get("crs_wkt")
                ds = ds.rio.write_crs(crs_wkt)

            except Exception as exc:
                raise ValueError(
                    "Unable to parse CRS from hydrologic input"
                ) from exc

            ds = self._reproject_match_to_dem(ds)
            return ds["velocity"].to_numpy()

        else:
            self._logger.info("Velocity not provided.")
            return None

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
        self._logger.info("Calculating dynamic cover variables.")

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
        ds_blh = utils.generate_pct_cover_custom(
            data_array=self.veg_type,
            veg_types=[15, 16, 17],  # these are the BLH zones + swamp
            x=8,
            y=8,
            boundary="pad",
        )

        ds_vegetated = utils.generate_pct_cover_custom(
            data_array=self.veg_type,
            veg_types=[
                v for v in range(15, 25)
            ],  # these are everything but open water
            x=8,
            y=8,
            boundary="pad",
        )

        self.pct_shrub_scrub = ds["pct_cover_12"].to_numpy()
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
        self.pct_blh = ds_blh.to_numpy()

    def _get_pct_pools(
        self,
        months: list[int],
        low: float = 3.0,
        high: float = 6.0,
    ):
        """Get percentage of "pool" pixels in each cell.

        Parameters
        -----------
        months : list
            List of months to average depth over
        low : float
            The lower end of the pools definition, defaults to 3m.
        high : float
            The high end of the pools definition, defaults to 6m.

        Returns
        -------
        pct_pools : np.ndarray
            480m numpy array of pct pools for given aggregation.
        """
        # subset water depth to time period
        ds = self.water_depth.sel(
            time=self.water_depth["time"].dt.month.isin(months)
        )
        da = ds.mean(dim="time", skipna=True)["height"]
        # mask to within bounds
        mask = (da >= low) & (da <= high)
        mask_float = mask.astype(float)
        pct_pools = mask_float.coarsen(x=8, y=8, boundary="pad").mean() * 100
        return pct_pools.values

    def _calculate_static_vars(self):
        """Get percent coverage variables for each 480m cell, based on 60m veg type pixels.
        This method is called only during initialization, for static variables that
        vary spatially but not temporally.
        """
        self._logger.info("Calculating static var: % developed and upland")
        self.pct_dev_upland = utils.generate_pct_cover_custom(
            data_array=self.initial_veg_type,
            # these are the dev'd (4) and upland (4)
            veg_types=[2, 3, 4, 5, 9, 10, 11, 12],
            x=8,
            y=8,
            boundary="pad",
        ).to_numpy()

        self._logger.info("Calculating static var: % flotant marsh")
        self.pct_flotant_marsh = utils.coarsen_and_reduce(
            da=self.flotant_marsh,
            veg_type=True,
            x=8,
            y=8,
            boundary="pad",
        ).to_numpy()

        self._logger.info("Calculating static var: crops")
        ds_crops = utils.generate_pct_cover_custom(
            data_array=self.initial_veg_type,  # use initial veg
            veg_types=[6, 7],
            x=8,
            y=8,
            boundary="pad",
        )
        self.pct_crops = ds_crops.to_numpy()

        self._logger.info("Calculating static var: developed")
        ds_developed = utils.generate_pct_cover_custom(
            data_array=self.initial_veg_type,  # use initial veg
            veg_types=[2, 3, 4, 5],
            x=8,
            y=8,
            boundary="pad",
        )
        self.pct_developed = ds_developed.to_numpy()

        self._logger.info("Calculating static var: pct area influence")
        self.human_influence = self._calculate_pct_area_influence(
            radius=self.testing_radius
        )

        surrounding_lu_data = self._calculate_surrounding_land_use()
        self.pct_forested_half_mi = surrounding_lu_data["forested"]
        self.pct_abandoned_ag_half_mi = surrounding_lu_data["abandoned_ag"]
        self.pct_pasture_half_mi = surrounding_lu_data["pasture"]
        self.pct_active_ag_water_half_mi = surrounding_lu_data[
            "active_ag_water"
        ]
        self.pct_nonhabitat_half_mi = surrounding_lu_data["nonhabitat"]

    @staticmethod
    def _calculate_near_landtype(
        landcover_arr: np.ndarray,
        landtype_true: list,
        radius: int,
        include_source: bool,
    ) -> np.ndarray:
        """Generalized method to get percentage of 480m cell with within
        radius of a land cover type or list of land cover types. Uses
        "initial vegtype" array, i.e. the initial conditions, with non-veg
        land cover classes included.

        Parameters
        -----------
        landtype : list
            list of land type int values considered "True", i.e. non-vegetated land.
            these are the "source" pixels.

        radius : int
            radius to use for the circular kernel, to determine nearby land cover

        include_source : bool
            if true, returns all pixels in the zone of influence (including source
            pixels). If false, source pixels (i.e. non forested) are not considered
            as within the zone.

        Returns
        --------
        istype_array : np.ndarray
            A boolean array where criteria is met
        """
        # make sure np array is used, w/ no Dask overhead from xarray
        landcover_arr = np.asarray(landcover_arr)
        istype_bool = np.isin(landcover_arr, landtype_true)
        nottype_bool = ~istype_bool

        disk_kernel = disk(radius)  # circular grid w/ radius (kernel)
        istype_expanded = binary_dilation(istype_bool, structure=disk_kernel)

        if include_source:
            return istype_expanded
        else:
            return istype_expanded & nottype_bool

    # def _calculate_pct_area_influence(self, radius: int = None) -> np.ndarray:
    #     """Percent of evaluation area inside of zones of influence defined by
    #     radii 5.7 km around towns; 3.5 km around cropland; and 1.1 km around
    #     residences.

    #     Radiuses are defined by circular (disk) kernel, which expands True
    #     pixels outward by r.

    #     5,700 / 60 = 95 pixels
    #     3,500 / 60 = 58.33 pixels

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     --------
    #     near_landtypes_da : np.ndarray
    #         Percent coverage array (480m grid cell)
    #     """
    #     towns = [2, 3, 4, 5]
    #     croplands = [6, 7, 8]
    #     radius_towns = radius or 95
    #     radius_croplands = radius or 59

    #     if radius:
    #         self._logger.warning(
    #             "Running area of influence with radius: %s.", radius
    #         )

    #     self._logger.info("Calculating static var: human influence - towns.")
    #     near_towns = self._calculate_near_landtype(
    #         landcover_arr=self.initial_veg_type,  # initial
    #         landtype_true=towns,
    #         radius=radius_towns,
    #         include_source=True,
    #     )
    #     self._logger.info(
    #         "Calculating static var: human influence - croplands."
    #     )
    #     near_croplands = self._calculate_near_landtype(
    #         landcover_arr=self.initial_veg_type,  # initial
    #         landtype_true=croplands,
    #         radius=radius_croplands,
    #         include_source=True,
    #     )

    #     stacked = np.stack([near_towns, near_croplands])
    #     influence_bool = np.any(stacked, axis=0)

    #     near_landtypes_da = xr.DataArray(influence_bool.astype(float))
    #     near_landtypes_da = (
    #         near_landtypes_da.coarsen(dim_0=8, dim_1=8, boundary="pad").mean()
    #         * 100
    #     )  # UNIT: index to pct

    #     return near_landtypes_da.to_numpy()

    def _calculate_pct_area_influence(self, radius: int = None) -> np.ndarray:
        """Percent of evaluation area inside of zones of influence defined by
        radii 5.7 km around towns; 3.5 km around cropland; and 1.1 km around
        residences.

        Radiuses are defined by circular (disk) kernel, which expands True
        pixels outward.

        480m radii:
        5,700 / 60 = 11.875 pixels
        3,500 / 60 = 7.29 pixels

        Parameters
        ----------
        Radius: int | None
            None if using model defaults, int if testing. Int also used during
            testing to speedup `HSI` execution time.

        Returns
        --------
        near_landtypes_da : np.ndarray
            Binary near landtypes array (480m grid cell)
        """
        crops_bool = self.pct_crops > 50
        developed_bool = self.pct_developed > 50

        disk_kernel = disk(radius or 7, strict_radius=False)
        crops_expanded = binary_dilation(
            crops_bool,
            structure=disk_kernel,
        )

        disk_kernel = disk(radius or 12, strict_radius=False)
        developed_expanded = binary_dilation(
            developed_bool,
            structure=disk_kernel,
        )

        stacked = np.stack([crops_expanded, developed_expanded])
        influence_bool = np.any(stacked, axis=0)
        return influence_bool

    def _calculate_surrounding_land_use(self):
        """
        Calculates buffered land use percentages, coarsens the results to the 480m,
        and return to a dictionary.
        """
        self._logger.info(
            "Calculating surrounding land use percentages (0.5-mile buffer)."
        )

        pct_dict_60m = utils.calculate_buffered_land_use_percentages(
            land_cover_da=self.initial_veg_type,
            resolution_m=60,
            buffer_miles=0.5,
        )

        forested = (
            pct_dict_60m["forested"]
            .coarsen(x=8, y=8, boundary="pad")
            .mean()
            .to_numpy()
        )

        abandoned_ag = (
            pct_dict_60m["abandoned_ag"]
            .coarsen(x=8, y=8, boundary="pad")
            .mean()
            .to_numpy()
        )

        pasture = (
            pct_dict_60m["pasture"]
            .coarsen(x=8, y=8, boundary="pad")
            .mean()
            .to_numpy()
        )

        active_ag_water = (
            pct_dict_60m["active_ag_water"]
            .coarsen(x=8, y=8, boundary="pad")
            .mean()
            .to_numpy()
        )

        nonhabitat = (
            pct_dict_60m["nonhabitat"]
            .coarsen(x=8, y=8, boundary="pad")
            .mean()
            .to_numpy()
        )

        return {
            "forested": forested,
            "abandoned_ag": abandoned_ag,
            "pasture": pasture,
            "active_ag_water": active_ag_water,
            "nonhabitat": nonhabitat,
        }

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
        self._logger.info(
            "Creating flotant marsh from initial veg and flotant marsh arrays."
        )
        da = xr.open_dataarray(self.flotant_marsh_path)
        da = da["band" == 0]
        # reproject to match hsi grid
        da = self._reproject_match_to_dem(da)

        # define which value is flotant marsh in raster key
        flotant_marsh = da == 4
        fresh_marsh = self.initial_veg_type["veg_type_subset"] == 20
        combined_flotant = flotant_marsh & fresh_marsh

        return combined_flotant

    # def _get_depth_filtered(
    #     self, months: None | list[int] = None
    # ) -> np.ndarray:
    #     """Calls the VegTransition _get_depth(), then adds a time
    #     filter (if supplied) and then resample to 480m cell size.

    #     Parameters
    #     ----------
    #     months : list (optional)
    #         List of months to average water depth over. If a list is not
    #         provided, the default is all months

    #     Return
    #     ------
    #     da_coarse : xr.DataArray
    #         A water depth data, averaged over a list of months (if provided)
    #         and then downscaled to 480m.
    #     """
    #     ds = super()._get_depth()  # VegTransition._get_depth()

    #     if not months:
    #         months = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

    #     filtered_ds = ds.sel(time=self.wse["time"].dt.month.isin(months))
    #     ds = filtered_ds.mean(dim="time", skipna=True)["height"]

    #     da_coarse = ds.coarsen(y=8, x=8, boundary="pad").mean()
    #     return da_coarse.to_numpy()

    def _get_daily_depth_filtered(
        self,
        months: None | list[int] = None,
        cell: bool = True,
    ) -> np.ndarray:
        """
        Reduce daily depth dataset to temporal mean, then resample to
        480m cell size.

        Parameters
        ----------
        months : list (optional)
            List of months to average water depth over. If a list is not
            provided, the default is all months

        cell : bool (optional)
            True if resampling input to 480m after temporal filter.
            Defaults to True.

        Return
        ------
        da_coarse : np.ndarray
            A water depth data, averaged over a list of months (or defaulting
            to one year) and then upscaled to 480m.
        """
        if not months:
            months = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

        filtered_ds = self.water_depth.sel(
            time=self.water_depth["time"].dt.month.isin(months)
        )
        da = filtered_ds.mean(dim="time", skipna=True)["height"]

        if cell is True:
            da = da.coarsen(y=8, x=8, boundary="pad").mean()

        return da.to_numpy()

    def _get_water_temperature_subset(
        self, months: None | list[int] = None, cell: bool = True
    ) -> np.ndarray:
        """
        Reduce daily water temperature dataset to temporal mean, then
        resample to 480m cell size.

        Parameters
        ----------
        months : list (optional)
            List of months to average water temp over. If a list is not
            provided, the default is all months

        cell : bool (optional)
            True if resampling input to 480m after temporal filter.
            Defaults to True.

        Return
        ------
        da_coarse : np.ndarray
            water temperature, averaged over a list of months (or defaulting
            to one year) and then upscaled to 480m (if cell=True) or kept at 60m (if cell=False).
        """
        if not months:
            months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        filtered_ds = self.water_temperature.sel(
            time=self.water_temperature["time"].dt.month.isin(months)
        )
        da = filtered_ds.mean(dim="time", skipna=True)["temperature"]

        if cell is True:
            da_coarse = da.coarsen(y=8, x=8, boundary="pad").mean()
            return da_coarse.to_numpy()
        else:
            return da.to_numpy()

    def _get_salinity_subset(
        self,
        months: None | list[int] = None,
        method: str = "mean",
    ) -> np.ndarray:
        """
        If salinity raster is prodided: reduce salinity dataset to temporal mean or max, then
        resample to 480m cell size. If no raster path is provided, create an
        approximate salinity array based on the habitat type.

        Parameters
        ----------
        months : list (optional)
            List of months to reduce salinity over. If a list is not
            provided, the default is all months

        method : str (optional)
            How to reduce. One of ["mean", "max", "upper-pctile-mean"].
            Defaults to mean. "upper-pctile-mean" is the mean of the
            upper 33% of values.

        Return
        ------
        da_coarse : np.ndarray
            salinity, averaged over a list of months (or defaulting
            to one year) and then upscaled to 480m.
        """
        if not months:
            months = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

        filtered_ds = self.salinity.sel(
            time=self.salinity["time"].dt.month.isin(months)
        )
        if method == "mean":
            da = filtered_ds.mean(dim="time", skipna=True)["salinity"]
        elif method == "max":
            da = filtered_ds.max(dim="time", skipna=True)["salinity"]
        elif method == "upper-pctile-mean":
            # 67th percentile (upper 33% is above this)
            threshold = filtered_ds.quantile(0.67)
            da = filtered_ds.where(filtered_ds >= threshold).mean(dim="time")[
                "salinity"
            ]

        da_coarse = da.coarsen(y=8, x=8, boundary="pad").mean()
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
        self._logger.info(
            "Calculating percent of canopy cover for mast types."
        )
        soft_mast = {"II": 0.53, "III": 0.61, "IV": 0.42, "V": 0.00}
        hard_mast = {"II": 0.0, "III": 0.39, "IV": 0.28, "V": 1.00}
        no_mast = {"II": 0.47, "III": 0.0, "IV": 0.3, "V": 0.0}

        # calculate as a weighted sum of zones
        self.pct_soft_mast = (
            self.pct_zone_ii * soft_mast["II"]
            + self.pct_zone_iii * soft_mast["III"]
            + self.pct_zone_iv * soft_mast["IV"]
            + self.pct_zone_v * soft_mast["V"]
        )
        self.pct_hard_mast = (
            self.pct_zone_ii * hard_mast["II"]
            + self.pct_zone_iii * hard_mast["III"]
            + self.pct_zone_iv * hard_mast["IV"]
            + self.pct_zone_v * hard_mast["V"]
        )

        self.pct_no_mast = (
            self.pct_zone_ii * no_mast["II"]
            + self.pct_zone_iii * no_mast["III"]
            + self.pct_zone_iv * no_mast["IV"]
            + self.pct_zone_v * no_mast["V"]
        )

        # pct coverage of either mast type
        self.pct_has_mast = self.pct_soft_mast + self.pct_hard_mast

    def _calculate_story_assignment(self):
        """Calculate categorical story assignment.

        Overstory = 3
        Midstory = 2
        Understory = 1
        """
        # initialize new, empty array for each timestep
        self.story_class = np.zeros_like(self.dem)
        self._logger.info("Calculating story assignment for forest types.")
        forested_types = [15, 16, 17, 18]
        understory_types = [20, 21, 22, 23]

        # overstory types
        type_mask = np.isin(self.veg_type.to_numpy(), forested_types)

        mask_3 = type_mask & (self.maturity > 10)
        self.story_class[mask_3] = 3

        # midstory types
        mask_2 = type_mask & (self.maturity < 10)
        self.story_class[mask_2] = 2

        # fresh_shrub
        mask_fresh_shrub = self.veg_type == 19
        self.story_class[mask_fresh_shrub] = 2

        # other
        type_mask = np.isin(self.veg_type.to_numpy(), understory_types)
        self.story_class[type_mask] = 1

        story_class_da = xr.DataArray(self.story_class, dims=["y", "x"])
        # get pct cover at 480m for swamp WVA
        self.pct_overstory = utils.generate_pct_cover_custom(
            data_array=story_class_da,
            veg_types=[3],
            x=8,
            y=8,
            boundary="pad",
        ).to_numpy()
        self.pct_midstory = utils.generate_pct_cover_custom(
            data_array=story_class_da,
            veg_types=[2],
            x=8,
            y=8,
            boundary="pad",
        ).to_numpy()
        self.pct_understory = utils.generate_pct_cover_custom(
            data_array=story_class_da,
            veg_types=[1],
            x=8,
            y=8,
            boundary="pad",
        ).to_numpy()

        # resample to 480m for BLH WVA
        self.story_class = utils.reduce_arr_by_mode(
            self.story_class
        ).to_numpy()

    def _calculate_shrub_scrub_midstory(self):
        """Get combined percentange of shrub/scrub & midstory cover at 480m cell size."""
        self.pct_shrub_scrub_midstory = (
            self.pct_midstory + self.pct_shrub_scrub
        )

    def _calculate_connectivity(self):
        """Calculate the size of contiguous forested areas in the domain, then
        bin into classed based on thresholds. This method uses 4-connectivity by default, but
        can be adapted to 8-connectivity. The output arrays contains zeros, for
        non-forested pixels (this may need to change).

        Note that this method uses a mode reduction to 480m, rather than mean reduction.

        Size is determined by 60m pixel area of 3,600m^2. Where:

        Category 1: < 5 acres = < 6 pixels
        Category 2: < 20 acres = < 23 pixels
        Category 3: < 100 acres = < 113 pixels
        Category 4: < 500 acres = < 562 pixels
        Category 5: > 500 acres = > 562 pixels
        """
        self._logger.info("Calculating connectivity for forested types.")
        forested_types = [15, 16, 17, 18]
        forest_bool = np.isin(self.veg_type, forested_types)

        # label all contiguous forest patches once
        labeled_array, num_features = label(forest_bool)

        # count pixels in each labeled region
        region_sizes = np.bincount(labeled_array.ravel())
        connectivity_category = np.zeros_like(labeled_array)

        for region_id, size in enumerate(region_sizes):
            if region_id == 0:
                continue  # skip 0
            if size <= 6:
                connectivity_category[labeled_array == region_id] = 1
            elif size <= 23:
                connectivity_category[labeled_array == region_id] = 2
            elif size <= 113:
                connectivity_category[labeled_array == region_id] = 3
            elif size <= 562:
                connectivity_category[labeled_array == region_id] = 4
            else:
                connectivity_category[labeled_array == region_id] = 5

        self.forested_connectivity_cat = utils.reduce_arr_by_mode(
            connectivity_category
        )

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
        self._logger.info("Calculating percent non-forested near forest.")
        forest_types = [15, 16, 17, 18]
        near_forest_mask = self._calculate_near_landtype(
            landcover_arr=self.initial_veg_type,
            landtype_true=forest_types,
            radius=radius,
            include_source=False,  # only non-forested pixels NEAR forested
        )

        near_forest_mask_da = xr.DataArray(near_forest_mask.astype(float))
        near_forest_mask_da = (
            near_forest_mask_da.coarsen(
                dim_0=8,
                dim_1=8,
                boundary="pad",
            ).mean()  # uses default coord names (dim_0, 1)
        ) * 100  # UNIT: index to percent

        self.pct_near_forest = near_forest_mask_da.to_numpy()

    def _load_blue_crab_lookup(self):
        """
        Read blue crab lookup table
        """
        self.blue_crab_lookup_table = pd.read_csv(self.blue_crab_lookup_path)

    def _create_output_file(self, resolution: int = 480):
        """HSI: Create NetCDF file for data output.

        Parameters
        ----------
        resolution : int
            Resolution in meters (480 or 60). Defaults to 480.

        Returns
        -------
        None
        """
        # Set filepath based on resolution
        if resolution == 480:
            self.netcdf_filepath = os.path.join(
                self.output_dir_path, f"{self.file_name}.nc"
            )
        elif resolution == 60:
            self.netcdf_filepath_60m = os.path.join(
                self.output_dir_path, f"{self.file_name}_60m.nc"
            )
        else:
            raise ValueError(f"Resolution must be 480 or 60, got {resolution}")

        # Load DEM as a template for coordinates
        da = xr.open_dataarray(self.dem_path)
        da = da.squeeze(drop="band")  # Drop 'band' dimension if it exists
        da = da.rio.write_crs("EPSG:6344")  # Assign CRS

        # Resample to target resolution if needed
        if resolution != 60:
            # Resample using rioxarray, with preserved correct coords and
            # assigns GeoTransform to spatial_ref. xr.coarsen() does not produce correct
            # projected coords.
            da = da.rio.reproject(
                da.rio.crs,
                resolution=resolution,
                resampling=Resampling.average,
            )

        # use coordinates at target resolution
        x = da.coords["x"].values
        y = da.coords["y"].values

        # Define the new time coordinate
        time_range = pd.date_range(
            start=f"{self.water_year_start}-10-01",
            end=f"{self.water_year_end}-10-01",
            freq="YS-OCT",  # Annual start
        )
        # encoding defined here and at append
        encoding = {
            "time": {
                "units": "days since 1850-01-01T00:00:00",
                "calendar": "gregorian",
                "dtype": "float64",
            }
        }

        ds = xr.Dataset(
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
            attrs={
                "title": f"HSI {resolution}m" if resolution == 60 else "HSI"
            },
        )

        ds = ds.rio.write_crs("EPSG:6344")

        # Save dataset to NetCDF with explicit encoding
        filepath = (
            self.netcdf_filepath_60m
            if resolution == 60
            else self.netcdf_filepath
        )
        ds.to_netcdf(filepath, encoding=encoding)
        self._logger.info(
            "Initialized %sm NetCDF file with CRS: %s", resolution, filepath
        )

    def _append_480m_hsi_vars_to_netcdf(self, timestep: pd.DatetimeTZDtype):
        """Append timestep data to the 480m NetCDF file.

        TODO: add warning if arrays are skipped because they are None
        TODO: move dict of arrays attrs to YAML or similar. It is too long
        to be inline now.

        Parameters
        ----------
        timestep : pd.DatetimeTZDtype
            Pandas datetime object of current timestep.

        Returns
        -------
        None
        """
        encoding = {
            "time": {
                "units": "days since 1850-01-01T00:00:00",
                "calendar": "gregorian",
                "dtype": "float64",
            }
        }

        timestep_str = timestep.strftime("%Y-%m-%d")
        hsi_variables = get_hsi_variables(self)

        with xr.open_dataset(self.netcdf_filepath, cache=False) as ds:
            ds_loaded = ds.load()  # loads into memory and closes file

        for var_name, (data, dtype, nc_attrs) in hsi_variables.items():
            if data is not None:  # only write arrays that have data
                netcdf_dtype = np.int8 if dtype == bool else dtype

                # if the var exists in the dataset, if not, initialize it
                if var_name not in ds_loaded:
                    shape = (
                        len(ds_loaded.time),
                        len(ds_loaded.y),
                        len(ds_loaded.x),
                    )
                    default_value = 0 if dtype == bool else np.nan
                    ds_loaded[var_name] = (
                        ["time", "y", "x"],
                        np.full(shape, default_value, dtype=netcdf_dtype),
                        nc_attrs,
                    )

                # boolean to int8 (0 and 1)
                if dtype == bool:
                    data = np.nan_to_num(data, nan=False).astype(np.int8)

                ds_loaded[var_name].loc[{"time": timestep}] = data.astype(
                    netcdf_dtype
                )

        # Save and close
        ds_loaded.to_netcdf(
            self.netcdf_filepath,
            mode="a",
            engine="h5netcdf",
            encoding=encoding,
        )
        ds_loaded.close()
        self._logger.info(
            "Appended timestep %s to 480m NetCDF file.", timestep_str
        )

    def _append_60m_hsi_vars_to_netcdf(self, timestep: pd.DatetimeTZDtype):
        """Append 60m QC variables to the 60m NetCDF file.

        Parameters
        ----------
        timestep : pd.DatetimeTZDtype
            Pandas datetime object of current timestep.

        Returns
        -------
        None
        """
        encoding = {
            "time": {
                "units": "days since 1850-01-01T00:00:00",
                "calendar": "gregorian",
                "dtype": "float64",
            }
        }

        timestep_str = timestep.strftime("%Y-%m-%d")

        # Define 60m QC variables to output
        qc_60m_variables = {
            "water_depth_july_august_mean": [
                self.water_depth_july_august_mean_60m,
                np.float32,
                {
                    "grid_mapping": "spatial_ref",
                    "units": "meters",
                    "long_name": "water depth July-August mean",
                    "description": (
                        "Mean water depth for July-August period at 60m resolution. "
                        "Used for pools/backwaters masking in catfish SI_5, "
                        "blackcrappie SI_2, SI_4, SI_8."
                    ),
                },
            ],
            "water_depth_july_sept_mean": [
                self.water_depth_july_sept_mean_60m,
                np.float32,
                {
                    "grid_mapping": "spatial_ref",
                    "units": "meters",
                    "long_name": "water depth July-September mean",
                    "description": (
                        "Mean water depth for July-September period at 60m resolution. "
                        "Used for pools/backwaters masking in catfish SI_8, SI_12, "
                        "SI_14, SI_18."
                    ),
                },
            ],
            "water_depth_may_july_mean": [
                self.water_depth_may_july_mean_60m,
                np.float32,
                {
                    "grid_mapping": "spatial_ref",
                    "units": "meters",
                    "long_name": "water depth May-July mean",
                    "description": (
                        "Mean water depth for May-July period at 60m resolution. "
                        "Used for pools/backwaters masking in catfish SI_10."
                    ),
                },
            ],
            "water_temperature_july_august_mean": [
                self.water_temperature_july_august_mean_60m,
                np.float32,
                {
                    "grid_mapping": "spatial_ref",
                    "units": "degrees_Celsius",
                    "long_name": "water temperature July-August mean",
                    "description": (
                        "Mean water temperature for July-August period at 60m resolution. "
                        "Used in catfish SI_5, blackcrappie SI_8, SI_9, SI_10."
                    ),
                },
            ],
            "water_temperature_july_sept_mean": [
                self.water_temperature_july_sept_mean_60m,
                np.float32,
                {
                    "grid_mapping": "spatial_ref",
                    "units": "degrees_Celsius",
                    "long_name": "water temperature July-September mean",
                    "description": (
                        "Mean water temperature for July-September period at 60m resolution. "
                        "Used in catfish SI_12, SI_14."
                    ),
                },
            ],
            "water_temperature_may_july_mean": [
                self.water_temperature_may_july_mean_60m,
                np.float32,
                {
                    "grid_mapping": "spatial_ref",
                    "units": "degrees_Celsius",
                    "long_name": "water temperature May-July mean",
                    "description": (
                        "Mean water temperature for May-July period at 60m resolution. "
                        "Used in catfish SI_10."
                    ),
                },
            ],
            "water_temperature_feb_march_mean": [
                self.water_temperature_feb_march_mean_60m,
                np.float32,
                {
                    "grid_mapping": "spatial_ref",
                    "units": "Deg C",
                    "long_name": "water temperature February-March mean",
                    "description": (
                        "Mean water temperature for February-March period at 480m resolution"
                    ),
                },
            ],
        }

        with xr.open_dataset(self.netcdf_filepath_60m, cache=False) as ds:
            ds_loaded = ds.load()  # loads into memory and closes file

        for var_name, (data, dtype, nc_attrs) in qc_60m_variables.items():
            if data is not None:  # only write arrays that have data
                netcdf_dtype = np.int8 if dtype == bool else dtype

                # if the var exists in the dataset, if not, initialize it
                if var_name not in ds_loaded:
                    shape = (
                        len(ds_loaded.time),
                        len(ds_loaded.y),
                        len(ds_loaded.x),
                    )
                    default_value = 0 if dtype == bool else np.nan
                    ds_loaded[var_name] = (
                        ["time", "y", "x"],
                        np.full(shape, default_value, dtype=netcdf_dtype),
                        nc_attrs,
                    )

                # boolean to int8 (0 and 1)
                if dtype == bool:
                    data = np.nan_to_num(data, nan=False).astype(np.int8)

                ds_loaded[var_name].loc[{"time": timestep}] = data.astype(
                    netcdf_dtype
                )

        # Save and close
        ds_loaded.to_netcdf(
            self.netcdf_filepath_60m,
            mode="a",
            engine="h5netcdf",
            encoding=encoding,
        )
        ds_loaded.close()
        self._logger.info(
            "Appended timestep %s to 60m NetCDF file.", timestep_str
        )

    def post_process(self):
        """HSI post process

        (1) Opens files and then crops to hydro domain
        (2) Create sidecar files with variables in the NetCDFs
        """
        # -------- crop 480m data --------
        self._logger.info("Post-processing 480m NetCDF file")
        with xr.open_dataset(self.netcdf_filepath) as ds:
            ds_out = ds.where(~np.isnan(self.hydro_domain_480)).copy(deep=True)
            # create sidecar info
            attrs_df_480 = utils.dataset_attrs_to_df(
                ds,
                selected_attrs=[
                    "long_name",
                    "description",
                    "units",
                ],
            )

            ds_out = ds_out.load()

        if os.path.exists(self.netcdf_filepath):
            os.remove(self.netcdf_filepath)

        ds_out.to_netcdf(self.netcdf_filepath, mode="w", engine="h5netcdf")

        # -------- create 480m sidecar file ---------
        self._logger.info("Creating 480m variable name text file")
        outpath = os.path.join(
            self.run_metadata_dir, f"{self.file_name}_hsi_netcdf_variables.csv"
        )
        attrs_df_480.to_csv(outpath, index=False)

        # -------- crop 60m data --------
        self._logger.info("Post-processing 60m NetCDF file")
        with xr.open_dataset(self.netcdf_filepath_60m) as ds:
            ds_out_60m = ds.where(~np.isnan(self.hydro_domain)).copy(deep=True)
            # create sidecar info
            attrs_df_60 = utils.dataset_attrs_to_df(
                ds,
                selected_attrs=[
                    "long_name",
                    "description",
                    "units",
                ],
            )

            ds_out_60m = ds_out_60m.load()

        if os.path.exists(self.netcdf_filepath_60m):
            os.remove(self.netcdf_filepath_60m)

        ds_out_60m.to_netcdf(
            self.netcdf_filepath_60m, mode="w", engine="h5netcdf"
        )

        # -------- create 60m sidecar file ---------
        self._logger.info("Creating 60m variable name text file")
        outpath_60m = os.path.join(
            self.run_metadata_dir,
            f"{self.file_name}_hsi_60m_netcdf_variables.csv",
        )
        attrs_df_60.to_csv(outpath_60m, index=False)

        self._logger.info("Post-processing complete.")

    def log_data_attribute_types(self):
        """Log the data type of all non-private attributes to help with debugging
        and understanding the current state of HSI variables.
        """
        self._logger.info("=== HSI Data Attribute Types ===")
        # non-private attributes, exclude those containing "path",
        # we just want to track data vars being supplied to the model
        attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("_") and "path" not in attr
        ]

        # dict of attribute types
        attr_types = {}
        for attr_name in sorted(attributes):
            try:
                attr_value = getattr(self, attr_name)
                attr_type = type(attr_value).__name__

                # Add shape info for arrays
                if hasattr(attr_value, "shape"):
                    attr_type += f"{attr_value.shape}"

                attr_types[attr_name] = attr_type

            except Exception as e:
                attr_types[attr_name] = f"Error: {e}"

        # log the full dictionary with pretty formatting
        formatted_dict = pprint.pformat(attr_types, width=100, indent=2)
        self._logger.info("HSI data inputs for run:\n%s", formatted_dict)


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
