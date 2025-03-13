from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class BaldEagleHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """
    hydro_domain_flag: bool # If True, all HSI SI arrays are masked to
    # hydro domain. If False, SI arrays relying only on veg type will maintain entire
    # veg type domain, which is a greate area then hydro domain.
    hydro_domain_480: np.ndarray = None
    dem_480: np.ndarray = None
    
    v1_pct_cell_developed_or_upland: np.ndarray = None
    v2_pct_cell_flotant_marsh: np.ndarray = None
    v3_pct_cell_forested_wetland: np.ndarray = None
    v4_pct_cell_fresh_marsh: np.ndarray = None
    v5_pct_cell_intermediate_marsh: np.ndarray = None
    v6_pct_cell_open_water: np.ndarray = None

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)
    si_3: np.ndarray = field(init=False)
    si_4: np.ndarray = field(init=False)
    si_5: np.ndarray = field(init=False)
    si_6: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create BaldEagleHSI instance from an HSI instance."""

        def safe_divide(array: np.ndarray, divisor: int = 100) -> np.ndarray:
            """Helper function to divide arrays when decimal values are required
            by the SI logic. In the case of None (no array) it is preserved and
            passed to SI methods."""
            return array / divisor if array is not None else None

        return cls(
            v1_pct_cell_developed_or_upland=safe_divide(
                hsi_instance.pct_dev_upland,
            ),
            v2_pct_cell_flotant_marsh=safe_divide(
                hsi_instance.pct_flotant_marsh,
            ),
            v3_pct_cell_forested_wetland=safe_divide(
                hsi_instance.pct_swamp_bottom_hardwood
            ),  # Note: "forested wetland" = BLH (lower, middle, upper) + swamp
            v4_pct_cell_fresh_marsh=safe_divide(
                hsi_instance.pct_fresh_marsh,
            ),
            v5_pct_cell_intermediate_marsh=safe_divide(
                hsi_instance.pct_intermediate_marsh
            ),
            v6_pct_cell_open_water=safe_divide(
                hsi_instance.pct_open_water,
            ),
            dem_480=hsi_instance.dem_480,
            hydro_domain_480 = hsi_instance.hydro_domain_480,
            hydro_domain_flag=hsi_instance.hydro_domain_flag
        )

    def __post_init__(self):
        """Run class methods to get HSI after instance is created."""
        # Set up the logger
        self._setup_logger()

        # Determine the shape of the arrays
        self.template = self._create_template_array()

        # Calculate individual suitability indices
        self.si_1 = self.calculate_si_1()
        self.si_2 = self.calculate_si_2()
        self.si_3 = self.calculate_si_3()
        self.si_4 = self.calculate_si_4()
        self.si_5 = self.calculate_si_5()
        self.si_6 = self.calculate_si_6()

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _create_template_array(self) -> np.ndarray:
        """Create an array from a template all valid pixels are 999.0, and
        NaN from the input are persisted.
        """
        # bald eagle does not have depth related vars, and is therfore not
        # limited to hyrologic model domain
        # arr = np.where(np.isnan(self.v2_water_depth_annual_mean), np.nan, 999.0)
        arr = np.full(self.v6_pct_cell_open_water.shape, 999.0)
        return arr

    def _setup_logger(self):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("BaldEagleHSI")
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

    def calculate_si_1(self) -> np.ndarray:
        """Percent of cell that is developed land or upland."""
        # Calculate for inital conditions and use for all time periods
        self._logger.info("Running SI 1")
        si_1 = self.template.copy()

        # if self.v1a_pct_cell_developed_land is None | self.v1b_pct_cell_upland is None:
        if self.v1_pct_cell_developed_or_upland is None:
            self._logger.info(
                "Pct developed land or upland data not provided. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else:
            # condition 1 (if no dev'd land or upland in cell)
            # note: we could just =0 vs !=0
            # mask_1 = self.v1a_pct_cell_developed_land | self.v1b_pct_cell_upland <= 0.0
            mask_1 = self.v1_pct_cell_developed_or_upland <= 0.0
            si_1[mask_1] = 0.01

            # condition 2 (otherwise)
            mask_2 = self.v1_pct_cell_developed_or_upland > 0.0
            si_1[mask_2] = 0.408 + 0.142 * np.log(
                self.v1_pct_cell_developed_or_upland[mask_2]
            )

            if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")
            
            if self.hydro_domain_flag:
                si_1 = np.where(~np.isnan(self.hydro_domain_480), si_1, np.nan)

        return si_1

    def calculate_si_2(self) -> np.ndarray:
        """Percent of cell that is flotant marsh."""
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()
        # Calculate for inital conditions and use for all time periods

        if self.v2_pct_cell_flotant_marsh is None:
            self._logger.info(
                "Pct flotant marsh data not provided. Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            # condition 1 (there is just one function here, so no need for mask)
            si_2 = (
                0.282
                + (0.047 * self.v2_pct_cell_flotant_marsh)
                - (1.105 * np.exp(-3) * self.v2_pct_cell_flotant_marsh**2)
                + (1.101 * np.exp(-5) * self.v2_pct_cell_flotant_marsh**3)
                - (3.967 * np.exp(-8) * self.v2_pct_cell_flotant_marsh**4)
            )

            if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")
            
            if self.hydro_domain_flag:
                si_2 = np.where(~np.isnan(self.hydro_domain_480), si_2, np.nan)

        return si_2

    def calculate_si_3(self) -> np.ndarray:
        """Percent of cell that is covered by forested wetland."""
        self._logger.info("Running SI 3")
        si_3 = self.template.copy()

        if self.v3_pct_cell_forested_wetland is None:
            self._logger.info(
                "Pct forested wetland data not provided. Setting index to 1."
            )
            si_3[~np.isnan(si_3)] = 1

        else:
            # condition 1 (there is just one function here, so no need for mask)
            si_3 = (
                0.015
                + (0.048 * self.v3_pct_cell_forested_wetland)
                - (1.178 * np.exp(-3) * self.v3_pct_cell_forested_wetland**2)
                + (1.366 * np.exp(-5) * self.v3_pct_cell_forested_wetland**3)
                - (5.673 * np.exp(-8) * self.v3_pct_cell_forested_wetland**4)
            )

            if np.any(np.isclose(si_3, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")
            
            if self.hydro_domain_flag:
                si_3 = np.where(~np.isnan(self.hydro_domain_480), si_3, np.nan)

        return si_3

    def calculate_si_4(self) -> np.ndarray:
        """Percent of cell that is covered by fresh marsh."""
        self._logger.info("Running SI 4")
        si_4 = self.template.copy()

        if self.v4_pct_cell_fresh_marsh is None:
            self._logger.info("Pct fresh marsh data not provided. Setting index to 1.")
            si_4[~np.isnan(si_4)] = 1

        else:
            # condition 1 (there is just one function here, so no need for mask)
            si_4 = (
                0.370
                + (0.07 * self.v4_pct_cell_fresh_marsh)
                - (2.655 * np.exp(-3) * self.v4_pct_cell_fresh_marsh**2)
                + (3.691 * np.exp(-5) * self.v4_pct_cell_fresh_marsh**3)
                - (1.701 * np.exp(-7) * self.v4_pct_cell_fresh_marsh**4)
            )

            if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")
            
            if self.hydro_domain_flag:
                si_4 = np.where(~np.isnan(self.hydro_domain_480), si_4, np.nan)

        return si_4

    def calculate_si_5(self) -> np.ndarray:
        """Percent of cell that is covered by intermediate marsh."""
        self._logger.info("Running SI 5")
        si_5 = self.template.copy()

        if self.v5_pct_cell_intermediate_marsh is None:
            self._logger.info(
                "Pct intermediate marsh data not provided. Setting index to 1."
            )
            si_5[~np.isnan(si_5)] = 1

        else:
            # condition 1 (there is just one function here, so no need for mask)
            si_5 = (
                0.263
                - (9.406 * np.exp(-3) * self.v5_pct_cell_intermediate_marsh)
                + (5.432 * np.exp(-4) * self.v5_pct_cell_intermediate_marsh**2)
                - (3.817 * np.exp(-6) * self.v5_pct_cell_intermediate_marsh**3)
            )

            if np.any(np.isclose(si_5, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")
            
            if self.hydro_domain_flag:
                si_5 = np.where(~np.isnan(self.hydro_domain_480), si_5, np.nan)

        return si_5

    def calculate_si_6(self) -> np.ndarray:
        """Percent of cell that is open water."""
        self._logger.info("Running SI 1")
        si_6 = self.template.copy()

        if self.v6_pct_cell_open_water is None:
            self._logger.info("Pct open water data not provided. Setting index to 1.")
            si_6[~np.isnan(si_6)] = 1

        else:
            # condition 1
            mask_1 = self.v6_pct_cell_open_water <= 0.0
            si_6[mask_1] = 0.01

            # condition 2 (AND)
            mask_2 = (self.v6_pct_cell_open_water > 0.0) & (
                self.v6_pct_cell_open_water <= 0.95
            )
            si_6[mask_2] = 0.985 - (0.105 * (self.v6_pct_cell_open_water[mask_2] ** -1))

            # condition 3
            mask_3 = self.v6_pct_cell_open_water > 0.95
            si_6[mask_3] = 0.0

            if np.any(np.isclose(si_6, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")
            
            if self.hydro_domain_flag:
                si_6 = np.where(~np.isnan(self.hydro_domain_480), si_6, np.nan)

        return si_6

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Bald Eagle final HSI.")
        for si_name, si_array in [
            ("SI 1", self.si_1),
            ("SI 2", self.si_2),
            ("SI 3", self.si_3),
            ("SI 4", self.si_4),
            ("SI 5", self.si_5),
            ("SI 6", self.si_6),
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
        hsi = (
            (self.si_1**0.0104)
            * (self.si_2**0.3715)
            * (self.si_3**0.4743)
            * (self.si_4**0.033)
            * (self.si_5**0.0353)
            * (self.si_6**0.0669)
        ) ** 0.991

        # Quality control check for invalid values: Ensure combined_score is between 0 and 1
        invalid_values = (hsi < 0) | (hsi > 1)
        if np.any(invalid_values):
            num_invalid = np.count_nonzero(invalid_values)
            self._logger.warning(
                "Combined suitability score has %d values outside [0,1]",
                num_invalid,
            )
            # Clip the combined_score to ensure it's between 0 and 1
            # hsi = np.clip(hsi, 0, 1)

        # subset final HSI array to vegetation domain (not hydrologic domain)
        # Masking: Set values in `mask` to NaN wherever `data` is NaN
        masked_hsi = np.where(np.isnan(self.dem_480), np.nan, hsi)

        return masked_hsi
