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
from datetime import datetime

import veg_logic
import hydro_logic


class VegTransition:
    """The Vegetation Transition Model.

    Vegetation zones are calculated independenlty (slower) for
    better model interperability and quality control.

    two dimensinal np.ndarray are used as default dtype, with xr.Dataset for cases
    where time series arrays are needed.
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

        # simulation
        self.start_date = config["simulation"].get("start_date")
        self.end_date = config["simulation"].get("end_date")

        # Time
        # self.time = 0

        # Store history for analysis
        # self.history = {"P": [self.P], "H": [self.H], "time": [self.time]}

        # Set up the logger
        self._setup_logger(log_level)

        # Load veg base and use as template to create arrays for the main state variables
        self.veg_type = self._load_veg_initial_raster()
        self.veg_keys = self._load_veg_keys()
        self.dem = self._load_dem()

        # load raster if provided, default values if not
        # self.load_salinity()

        self.wse = None
        self.maturity = np.ones_like(self.dem)
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
        self._logger.info("starting timestap: %s", date)
        wy = date.year
        # copy existing veg types
        veg_type_in = self.veg_type

        # calculate depth
        self.wse = self._load_wse_wy(wy, variable_name="WSE_MEAN")
        self.wse = self._reproject_match_to_dem(self.wse)  # TEMPFIX
        self.water_depth = self._get_depth()

        # veg_type array is iteratively updated, for each zone
        # self.veg_type = veg_logic.zone_v(self.veg_type, self.water_depth)
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

    def run(self, start_date, end_date):
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

        for timestep in pd.date_range(self.start_date, self.end_date, freq="y"):
            self.step(timestep)

        self._logger.info("Simulation complete")

    def _load_dem(self) -> xr.Dataset:
        """Load project domain DEM."""
        ds = xr.open_dataset(self.dem_path)
        ds = ds.squeeze(drop="band_data")
        da = ds.to_dataarray(dim="band")
        self._logger.info("Loaded DEM")
        return da.to_numpy()

    # def _load_wse_timestep(
    #     self,
    #     date: str,
    #     variable_name: str = "WSE_MEAN",
    #     date_format: str = "%Y_%m_%d",
    # ) -> Optional[xr.DataArray]:
    #     """
    #     Load a single timestep from a folder of .tif files as an xarray.DataArray.

    #     The .tif file should have a filename that includes a variable name (e.g., `WSE_MEAN`)
    #     followed by a timestamp in the specified format (e.g., `2005_10_01`). The function will
    #     locate the file corresponding to the specified date and load it as a DataArray.

    #     Parameters
    #     ----------
    #     folder_path : str
    #         Path to the folder containing the .tif files.
    #     date : str
    #         The date for the timestep to load, formatted according to `date_format`.
    #     variable_name : str, optional
    #         The name of the variable to use in the DataArray.
    #     date_format : str, optional
    #         Format string for parsing dates from file names, default is "%Y_%m_%d".
    #         Adjust based on your file naming convention.

    #     Returns
    #     -------
    #     xr.DataArray or None
    #         An xarray.DataArray with the raster data for the specified timestep,
    #         or None if the file is not found.
    #     """
    #     # Format the date into the specified date_format to match the file names
    #     date_str = pd.to_datetime(date).strftime(date_format)

    #     # Construct the expected filename pattern
    #     expected_filename = f"{variable_name}_{date_str}.tif"
    #     file_path = os.path.join(self.wse_directory_path, expected_filename)

    #     # Check if the file exists
    #     if not os.path.exists(file_path):
    #         self._logger.info("File not found: %s", file_path)
    #         return None

    #     # Load the .tif file as a DataArray
    #     da = rioxarray.open_rasterio(file_path).squeeze(dim="band")

    #     # Rename the variable and add the time coordinate
    #     da = da.rename(variable_name).expand_dims(time=[pd.to_datetime(date)])

    #     return da

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
        # tif_files = sorted(glob.glob(os.path.join(self.wse_directory_path, "*.tif")))
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

        return xr_dataset

    def _reproject_match_to_dem(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Temporary fix to match WSE model output to 60m DEM grid.
        """
        ds_dem = xr.open_dataset(self.dem_path)
        ds_dem = ds_dem.squeeze(drop="band_data")
        da_dem = ds_dem.to_dataarray(dim="band")

        self._logger.warning("reprojecting WSE to match DEM. TEMPFIX!")
        return ds.rio.reproject_match(da_dem)

    def _get_depth(self) -> xr.Dataset:
        """Calculate water depth from DEM and Water Surface Elevation.

        TODO: update to work for all timesteps in Dataset
        """
        self._logger.info("Creating depth")
        return self.wse - self.dem

    def _calculate_maturity(self, veg_type_in: np.ndarray):
        """
        +1 maturity for pixels without vegetation changes.
        """
        # TODO: Need to create mask for only Zone II to V pixels.
        self._logger.warning("need to mask to zone II to V pixels.")

        # Ensure both arrays have the same shape
        if veg_type_in.shape != self.veg_type.shape:
            raise ValueError("Input arrays must have the same shape.")

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
        return da.to_numpy()

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
