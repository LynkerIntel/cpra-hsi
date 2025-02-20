from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class AlligatorHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    # gridded data as numpy arrays or None
    # init with None to be distinct from np.nan
    v1_pct_open_water: np.ndarray = None
    v2_water_depth_annual_mean: np.ndarray = None

    # v3_pct_cell_covered_by_habitat_types: np.ndarray = None
    v3a_pct_swamp_bottom_hardwood: np.ndarray = None
    v3b_pct_fresh_marsh: np.ndarray = None
    v3c_pct_intermediate_marsh: np.ndarray = None
    v3d_pct_brackish_marsh: np.ndarray = None

    v4_edge: np.ndarray = None
    v5_mean_annual_salinity: np.ndarray = None

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)
    si_3: np.ndarray = field(init=False)
    si_4: np.ndarray = field(init=False)
    si_5: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create AlligatorHSI instance from an HSI instance."""
        return cls(
            v1_pct_open_water=hsi_instance.pct_open_water / 100,
            v2_water_depth_annual_mean=hsi_instance.water_depth_annual_mean,
            v3a_pct_swamp_bottom_hardwood=hsi_instance.pct_swamp_bottom_hardwood / 100,
            v3b_pct_fresh_marsh=hsi_instance.pct_fresh_marsh / 100,
            v3c_pct_intermediate_marsh=hsi_instance.pct_intermediate_marsh / 100,
            v3d_pct_brackish_marsh=hsi_instance.pct_brackish_marsh / 100,
            v4_edge=hsi_instance.edge,
            v5_mean_annual_salinity=hsi_instance.mean_annual_salinity,
        )

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
        self.si_5 = self.calculate_si_5()

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _setup_logger(self):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("AlligatorHSI")
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
        return self.v1_pct_open_water.shape
        # raise ValueError("At least one S.I. raster input must be provided.")

    def calculate_si_1(self) -> np.ndarray:
        """Percent of cell that is open water."""
        if self.v1_pct_open_water is None:
            self._logger.info("Pct open water data not provided. Setting index to 1.")
            si_1 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 1")
            # Create an array to store the results
            si_1 = np.full(self._shape, 999.0)

            # condition 1
            mask_1 = self.v1_pct_open_water < 0.2
            si_1[mask_1] = (4.5 * self.v1_pct_open_water[mask_1]) + 0.1

            # condition 2
            mask_2 = (self.v1_pct_open_water >= 0.2) & (self.v1_pct_open_water <= 0.4)
            si_1[mask_2] = 1

            # condition 3
            mask_3 = self.v1_pct_open_water > 0.4
            si_1[mask_3] = (-1.667 * self.v1_pct_open_water[mask_3]) + 1.667

            # Check for unhandled condition with tolerance
            if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

        return si_1

    def calculate_si_2(self) -> np.ndarray:
        """Mean annual water depth relative to the marsh surface."""
        if self.v2_water_depth_annual_mean is None:
            self._logger.info(
                "avg annual water depth relative to marsh surface data not provided. Setting index to 1."
            )
            si_2 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 2")
            # Not using 999 as init value, because depth has
            # many NaN pixels, instead these become 0
            si_2 = np.full(self._shape, 0.0)  # must be float value!

            # condition 1 (OR)
            mask_1 = (self.v2_water_depth_annual_mean <= -0.55) | (
                self.v2_water_depth_annual_mean >= 0.25
            )
            si_2[mask_1] = 0.1

            # condition 2 (AND)
            mask_2 = (self.v2_water_depth_annual_mean >= -0.55) & (
                self.v2_water_depth_annual_mean <= 0.15
            )
            si_2[mask_2] = (2.25 * self.v2_water_depth_annual_mean[mask_2]) + 1.3375

            # condition 3 (AND)
            mask_3 = (self.v2_water_depth_annual_mean > -0.15) & (
                self.v2_water_depth_annual_mean < 0.25
            )
            si_2[mask_3] = (-2.25 * self.v2_water_depth_annual_mean[mask_3]) + 0.6625

            # Check for unhandled condition with tolerance
            if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

        return si_2

    def calculate_si_3(self) -> np.ndarray:
        """Proportion of cell covered by habitat types."""
        self._logger.info("Running SI 3")

        for array in [
            self.v3a_pct_swamp_bottom_hardwood,
            self.v3b_pct_fresh_marsh,
            self.v3c_pct_intermediate_marsh,
            self.v3d_pct_brackish_marsh,
        ]:
            if array is None:
                self._logger.info("%s not provided. Setting index to 1.", array)
                array = np.ones(self._shape)

        si_3 = (
            (0.551 * self.v3a_pct_swamp_bottom_hardwood)
            + (0.713 * self.v3b_pct_fresh_marsh)
            + (1.0 * self.v3c_pct_intermediate_marsh)
            + (0.408 * self.v3d_pct_brackish_marsh)
        )

        # TODO: error handling here? (for case where no blank arr is initialized)
        return si_3

    def calculate_si_4(self) -> np.ndarray:
        """Edge."""
        if self.v4_edge is None:
            self._logger.info("Edge data not provided. Setting index to 1.")
            si_4 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 4")
            si_4 = np.full(self._shape, 999.0)
            mask_1 = (self.v4_edge >= 0) & (self.v4_edge <= 22)
            si_4[mask_1] = 0.05 + (0.95 * (self.v4_edge[mask_1] / 22))

            mask_2 = self.v4_edge > 22
            si_4[mask_2] = 1

            # Check for unhandled condition with tolerance
            if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

        return si_4

    def calculate_si_5(self) -> np.ndarray:
        """Mean annual salinity."""
        if self.v5_mean_annual_salinity is None:
            self._logger.info(
                "mean annual salinity data not provided. Setting index to 1."
            )
            si_5 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 5")
            si_5 = np.full(self._shape, 999.0)  # must be float
            mask_1 = (self.v5_mean_annual_salinity >= 0) & (
                self.v5_mean_annual_salinity <= 10
            )

            si_5[mask_1] = (-0.1 * self.v5_mean_annual_salinity[mask_1]) + 1

            mask_2 = self.v5_mean_annual_salinity > 10
            si_5[mask_2] = 0

            # Check for unhandled condition with tolerance
            if np.any(np.isclose(si_5, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

        return si_5

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Alligator final HSI.")
        for si_name, si_array in [
            ("SI 1", self.si_1),
            ("SI 2", self.si_2),
            ("SI 3", self.si_3),
            ("SI 4", self.si_4),
            ("SI 5", self.si_5),
        ]:
            invalid_values = (si_array < 0) | (si_array > 1)
            if np.any(invalid_values):
                num_invalid = np.count_nonzero(invalid_values)
                self._logger.warning(
                    "%s contains %d values outside the range [0, 1].",
                    si_name,
                    num_invalid,
                )

        # Combine individual suitability indices
        hsi = (self.si_1 * self.si_2 * self.si_3 * self.si_4 * self.si_5) ** (1 / 5)

        # Check the final HSI array for invalid values
        invalid_values_hsi = (hsi < 0) | (hsi > 1)
        if np.any(invalid_values_hsi):
            num_invalid_hsi = np.count_nonzero(invalid_values_hsi)
            self._logger.warning(
                "Final HSI contains %d values outside the range [0, 1].",
                num_invalid_hsi,
            )

        return hsi
