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

    # hydro domain. If False, SI arrays relying only on veg type will maintain entire
    # veg type domain, which is a greate area then hydro domain.
    hydro_domain_480: np.ndarray = None
    dem_480: np.ndarray = None

    # gridded data as numpy arrays or None
    # init with None to be distinct from np.nan
    v1_mean_annual_salinity: np.ndarray = None
    v2_mean_water_depth_jan_aug: np.ndarray = None
    # v3_pct_cell_covered_by_habitat_types: np.ndarray = None
    v3a_pct_cell_swamp_bottomland_hardwood: np.ndarray = None
    v3b_pct_cell_fresh_marsh: np.ndarray = None
    v3c_pct_cell_open_water: np.ndarray = None
    v3d_pct_cell_intermediate_marsh: np.ndarray = None
    v3e_pct_cell_brackish_marsh: np.ndarray = None
    v3f_pct_cell_saline_marsh: np.ndarray = None
    v3g_pct_cell_bare_ground: np.ndarray = None
    v4_mean_water_depth_sept_dec: np.ndarray = None

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)
    si_3: np.ndarray = field(init=False)
    si_4: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create CrawfishHSI instance from an HSI instance.

        All HSI depth vars in provided in m. Necessary conversions take here,
        with care to maintain None vars.
        """

        def safe_divide(array: np.ndarray, divisor: int = 100) -> np.ndarray:
            """Helper function to divide arrays when decimal values are required
            by the SI logic. In the case of None (no array) it is preserved and
            passed to SI methods."""
            return array / divisor if array is not None else None

        def safe_multiply(
            array: np.ndarray, multiplier: int = 100
        ) -> np.ndarray:
            """Helper function to multiply arrays when cm values are required
            by the SI logic. In the case of None (no array) it is preserved and
            passed to SI methods."""
            return array * multiplier if array is not None else None

        return cls(
            v1_mean_annual_salinity=hsi_instance.mean_annual_salinity,
            v2_mean_water_depth_jan_aug=safe_multiply(
                hsi_instance.water_depth_monthly_mean_jan_aug_cm
            ),
            v3a_pct_cell_swamp_bottomland_hardwood=safe_divide(
                hsi_instance.pct_swamp_bottom_hardwood
            ),
            v3b_pct_cell_fresh_marsh=safe_divide(hsi_instance.pct_fresh_marsh),
            v3c_pct_cell_open_water=safe_divide(hsi_instance.pct_open_water),
            v3d_pct_cell_intermediate_marsh=safe_divide(
                hsi_instance.pct_intermediate_marsh
            ),
            v3e_pct_cell_brackish_marsh=safe_divide(
                hsi_instance.pct_brackish_marsh
            ),
            v3f_pct_cell_saline_marsh=safe_divide(
                hsi_instance.pct_saline_marsh
            ),
            v3g_pct_cell_bare_ground=safe_divide(hsi_instance.pct_bare_ground),
            v4_mean_water_depth_sept_dec=safe_multiply(
                hsi_instance.water_depth_monthly_mean_sept_dec
            ),
            dem_480=hsi_instance.dem_480,
            hydro_domain_480=hsi_instance.hydro_domain_480,
        )

    def __post_init__(self):
        """Run class methods to get HSI after instance is created."""
        # Set up the logger
        self._setup_logger()

        self.template = self._create_template_array()

        # Calculate individual suitability indices
        self.si_1 = self.calculate_si_1()
        self.si_2 = self.calculate_si_2()
        self.si_3 = self.calculate_si_3()
        self.si_4 = self.calculate_si_4()

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _create_template_array(self) -> np.ndarray:
        """Create an array from a template all valid pixels are 999.0, and
        NaN from the input are persisted.
        """
        arr = np.where(
            np.isnan(self.v2_mean_water_depth_jan_aug), np.nan, 999.0
        )
        return arr

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

    def calculate_si_1(self) -> np.ndarray:
        """Mean annual salinity."""
        self._logger.info("Running SI 1")
        si_1 = self.template.copy()

        if self.v1_mean_annual_salinity is None:
            self._logger.info(
                "mean annual salinity data not provided. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else:
            # condition 1
            mask_1 = self.v1_mean_annual_salinity <= 1.5
            si_1[mask_1] = 1.0

            # condition 2 (AND)
            mask_2 = (self.v1_mean_annual_salinity > 1.5) & (
                self.v1_mean_annual_salinity <= 3.0
            )
            si_1[mask_2] = 1.5 - (0.333 * self.v1_mean_annual_salinity[mask_2])

            # condition 3 (AND)
            mask_3 = (self.v1_mean_annual_salinity > 3.0) & (
                self.v1_mean_annual_salinity <= 6.0
            )
            si_1[mask_3] = 1.0 - (0.167 * self.v1_mean_annual_salinity[mask_3])

            # condition 4
            mask_4 = self.v1_mean_annual_salinity > 6.0
            si_1[mask_4] = 0.0

            if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

            # if self.hydro_domain_flag:
            #     si_1 = np.where(~np.isnan(self.hydro_domain_480), si_1, np.nan)

        return si_1

    def calculate_si_2(self) -> np.ndarray:
        """Mean water depth from January to August in cm.

        Logic is defined in cm, and data should be provided in cm.
        """
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()

        if self.v2_mean_water_depth_jan_aug is None:
            self._logger.info(
                "mean water depth from january to august data not provided. Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            # condition 1 (OR)
            mask_1 = (self.v2_mean_water_depth_jan_aug <= 0.0) | (
                self.v2_mean_water_depth_jan_aug > 274  # RHS is in meters
            )
            si_2[mask_1] = 0.0

            # condition 2 (AND)
            mask_2 = (self.v2_mean_water_depth_jan_aug > 0.0) & (
                self.v2_mean_water_depth_jan_aug <= 46
            )
            si_2[mask_2] = 0.02174 * self.v2_mean_water_depth_jan_aug[mask_2]

            # condition 3 (AND)
            mask_3 = (self.v2_mean_water_depth_jan_aug > 46) & (
                self.v2_mean_water_depth_jan_aug <= 91
            )
            si_2[mask_3] = 1.0

            # condition 4 (AND)
            mask_4 = (self.v2_mean_water_depth_jan_aug > 91) & (
                self.v2_mean_water_depth_jan_aug <= 274
            )
            si_2[mask_4] = 1.5 - (
                0.00457 * self.v2_mean_water_depth_jan_aug[mask_4]
            )

            if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

            # if self.hydro_domain_flag:
            #     si_2 = np.where(~np.isnan(self.hydro_domain_480), si_2, np.nan)

        return si_2

    def calculate_si_3(self) -> np.ndarray:
        """Proportion of cell covered by habitat types."""
        self._logger.info("Running SI 3")
        si_3 = self.template.copy()

        si_3 = (
            (1.0 * self.v3a_pct_cell_swamp_bottomland_hardwood)
            + (0.85 * self.v3b_pct_cell_fresh_marsh)
            + (0.75 * self.v3c_pct_cell_open_water)
            + (0.6 * self.v3d_pct_cell_intermediate_marsh)
            + (0.2 * self.v3e_pct_cell_brackish_marsh)
            + (0.0 * self.v3f_pct_cell_saline_marsh)
            + (0.0 * self.v3g_pct_cell_bare_ground)
        )

        if np.any(np.isclose(si_3, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        # if self.hydro_domain_flag:
        #         si_3 = np.where(~np.isnan(self.hydro_domain_480), si_3, np.nan)

        return si_3

    def calculate_si_4(self) -> np.ndarray:
        """Mean water depth from September to December in cm.

        Logic is defined in cm, and data should be provided in cm.
        """
        self._logger.info("Running SI 4")
        si_4 = self.template.copy()

        if self.v4_mean_water_depth_sept_dec is None:
            self._logger.info(
                "mean water depth from september to december data not provided. Setting index to 1."
            )
            si_4[~np.isnan(si_4)] = 1

        else:
            # condition 1
            mask_1 = (
                self.v4_mean_water_depth_sept_dec <= 0.0
            )  # RHS is in meters
            si_4[mask_1] = 1.0

            # condition 2 (AND)
            mask_2 = (self.v4_mean_water_depth_sept_dec > 0.0) & (
                self.v4_mean_water_depth_sept_dec <= 15
            )
            si_4[mask_2] = 1.0 - (
                0.06667 * self.v4_mean_water_depth_sept_dec[mask_2]
            )

            # condition 3
            mask_3 = self.v4_mean_water_depth_sept_dec > 15
            si_4[mask_3] = 0.0

            if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

            # if self.hydro_domain_flag:
            #     si_4 = np.where(~np.isnan(self.hydro_domain_480), si_4, np.nan)

        return si_4

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Crayfish final HSI.")
        for si_name, si_array in [
            ("SI 1", self.si_1),
            ("SI 2", self.si_2),
            ("SI 3", self.si_3),
            ("SI 4", self.si_4),
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
            ((self.si_1 * self.si_2) ** (1 / 6))
            * (self.si_3 ** (1 / 3))
            * (self.si_4 ** (1 / 3))
        )

        # Quality control check: Ensure combined_score is between 0 and 1
        invalid_values = (hsi < 0) | (hsi > 1)
        if np.any(invalid_values):
            num_invalid = np.count_nonzero(invalid_values)
            self._logger.warning(
                "Combined suitability score has %d values outside [0,1].",
                num_invalid,
            )

        # subset final HSI array to vegetation domain (not hydrologic domain)
        # Masking: Set values in `mask` to NaN wherever `data` is NaN
        masked_hsi = np.where(np.isnan(self.dem_480), np.nan, hsi)

        return masked_hsi
