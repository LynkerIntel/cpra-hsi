from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class CrawfishHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    # gridded data as numpy arrays or None
    # init with None to be distinct from np.nan
    v1_mean_annual_salinity: np.ndarray = None
    v2_mean_water_depth_jan_aug: np.ndarray = None
    v3_pct_cell_covered_by_habitat_types: np.ndarray = None
    v4_mean_water_depth_sept_dec: np.ndarray = None

    # Species-specific parameters (example values)
    # for cases where static values are used

    # optimal_temperature: float = 20.0  # EXAMPLE
    # temperature_tolerance: float = 10.0  # EXAMPLE

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)
    si_3: np.ndarray = field(init=False)
    si_4: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

    def __post_init__(self):
        """Run class methods to get HSI after instance is created."""
        # Set up the logger
        self._setup_logger()

        # Determine the shape of the arrays
        self._shape = self._determine_shape()

        # Calculate individual suitability indices
        self.si_1 = self.calculate_si_1()
        self.si_2 = self.calculate_si_2()
        self.si_3 = self.calculate_si_3()
        self.si_4 = self.calculate_si_4()

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _setup_logger(self):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("CrawfishHSI")
        self._logger.setLevel(logging.INFO)

        # Prevent adding multiple handlers if already added
        if not self._logger.handlers:
            # Create console handler and set level
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Create formatter and add it to the handler
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)

            # Add the handler to the logger
            self._logger.addHandler(ch)

    def _determine_shape(self) -> tuple:
        """Determine the shape of the environmental variable arrays."""
        # Iterate over instance attributes and return the shape of the first non None numpy array
        for name, value in vars(self).items():
            if value is not None and isinstance(value, np.ndarray):
                self._logger.info(
                    "Using attribute %s as shape for output: %s", name, value.shape
                )
                return value.shape

        raise ValueError("At least one S.I. raster input must be provided.")

    def calculate_si_1(self) -> np.ndarray:
        """Mean annual salinity."""
        if self.v1_mean_annual_salinity is None:
            self._logger.info("mean annual salinity data not provided. Setting index to 1.")
            si_1 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 1")
            # Create an array to store the results
            si_1 = np.full(self._shape, 999)

            # condition 1
            mask_1 = self.v1_mean_annual_salinity <= 1.5
            si_1[mask_1] = 1.0

            # condition 2 (AND)
            mask_2 = (self.v1_mean_annual_salinity > 1.5) & (self.v1_mean_annual_salinity <= 3.0)
            si_1[mask_2] = 1.5 - (0.333 * self.v1_mean_annual_salinity[mask_2])

            # condition 3 (AND)
            mask_3 = (self.v1_mean_annual_salinity > 3.0) & (self.v1_mean_annual_salinity <= 6.0)
            si_1[mask_3] = 1.0 - (0.167 * self.v1_mean_annual_salinity[mask_3])

            # condition 4
            mask_4 = self.v1_mean_annual_salinity > 6.0
            si_1[mask_4] = 0.0

            if 999 in si_1:
                raise ValueError("Unhandled condition in SI logic!")

        return si_1

    def calculate_si_2(self) -> np.ndarray:
        """Mean water depth from January to August in cm."""
        if self.v2_mean_water_depth_jan_aug is None:
            self._logger.info(
                "mean water depth from january to august data not provided. Setting index to 1."
            )
            si_2 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 2")
            si_2 = np.full(self._shape, 999)

            # condition 1 (OR)
            mask_1 = (self.v2_mean_water_depth_jan_aug <= 0.0) | (
                self.v2_mean_water_depth_jan_aug > 274.0 #RHS is in cm
            )
            si_2[mask_1] = 0.0

            # condition 2 (AND)
            mask_2 = (self.v2_mean_water_depth_jan_aug > 0.0) & (
                self.v2_mean_water_depth_jan_aug <= 46.0
            )
            si_2[mask_2] = 0.02174 * self.v2_mean_water_depth_jan_aug[mask_2]

            # condition 3 (AND)
            mask_3 = (self.v2_mean_water_depth_jan_aug > 46.0) & (
                self.v2_mean_water_depth_jan_aug <= 91.0
            )
            si_2[mask_3] = 1.0

            # condition 4 (AND)
            mask_4 = (self.v2_mean_water_depth_jan_aug > 91.0) & (
                self.v2_mean_water_depth_jan_aug <= 274.0
            )
            si_2[mask_4] = 1.5 - (0.00457 * self.v2_mean_water_depth_jan_aug[mask_4])

            if 999 in si_2:
                raise ValueError("Unhandled condition in SI logic!")

        return si_2

    # JG TODO 10.31.24: not yet fully implimented 
    def calculate_si_3(self) -> np.ndarray:
        """Proportion of cell covered by habitat types."""
        if self.v3_pct_cell_covered_by_habitat_types is None:
            self._logger.info(
                "Pct habitat types data not provided. Setting index to 1."
            )
            si_3 = np.ones(self._shape)

        else:
            return NotImplementedError

        return si_3

 def calculate_si_4(self) -> np.ndarray:
        """Mean water depth from September to December in cm."""
        if self.v4_mean_water_depth_sept_dec is None:
            self._logger.info(
                "mean water depth from september to december data not provided. Setting index to 1."
            )
            si_4 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 4")
            si_4 = np.full(self._shape, 999)

            # condition 1
            mask_1 = (self.v4_mean_water_depth_sept_dec <= 0.0) #RHS is in cm
            si_4[mask_1] = 1.0

            # condition 2 (AND)
            mask_2 = (self.v4_mean_water_depth_sept_dec > 0.0) & (
                self.v4_mean_water_depth_sept_dec <= 15.0
            )
            si_4[mask_2] = 1.0 - (0.06667 * self.v4_mean_water_depth_sept_dec[mask_2])

            # condition 3 
            mask_3 = (self.v4_mean_water_depth_sept_dec > 15.0)
            si_4[mask_3] = 0.0

            if 999 in si_4:
                raise ValueError("Unhandled condition in SI logic!")

        return si_4
    

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        hsi = ((self.si_1 * self.si_2) ** 1/6) * (self.si_3 ** 1/3) * (self.si_4 ** 1/3)

        # Quality control check: Ensure combined_score is between 0 and 1
        invalid_values = (hsi < 0) | (hsi > 1)
        if np.any(invalid_values):
            num_invalid = np.count_nonzero(invalid_values)
            self._logger.warning(
                "Combined suitability score has %d values outside [0,1]. Clipping values.",
                num_invalid,
            )
            # Clip the combined_score to ensure it's between 0 and 1
            # hsi = np.clip(hsi, 0, 1)

        return hsi