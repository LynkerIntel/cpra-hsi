from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class GizzardShadHSI:
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
    v1_tds_summer_growing_season: np.ndarray = None  # ideal
    v2_avg_num_frost_free_days_growing_season: np.ndarray = None  # ideal
    v3_mean_weekly_summer_temp: np.ndarray = None  # ideal
    v4_max_do_summer: np.ndarray = None  # ideal
    v5a_water_lvl_change: np.ndarray = None  # ideal
    v5b_is_veg_inundated: np.ndarray = None #ideal
    v6_mean_weekly_temp_reservoir_spawning_season: np.ndarray = None  # ideal
    v7a_pct_vegetated: np.ndarray = None
    v7b_water_depth_spawning_season: np.ndarray = None

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)
    si_3: np.ndarray = field(init=False)
    si_4: np.ndarray = field(init=False)
    si_5: np.ndarray = field(init=False)
    si_6: np.ndarray = field(init=False)
    si_7: np.ndarray = field(init=False)

    
    # Set up components and equations
    food_component: np.ndarray = field(init=False)
    water_quality: np.ndarray = field(init=False)
    reproduction: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create GizzardshadHSI instance from an HSI instance."""
        return cls(
            v1_tds_summer_growing_season=hsi_instance.tds_summer_growing_season,
            v2_avg_num_frost_free_days_growing_season=hsi_instance.avg_num_frost_free_days_growing_season,
            v3_mean_weekly_summer_temp=hsi_instance.mean_weekly_summer_temp,
            v4_max_do_summer=hsi_instance.max_do_summer,
            v5a_water_lvl_change=hsi_instance.water_lvl_change,
            v5b_is_veg_inundated=hsi_instance.is_veg_inundated,
            v6_mean_weekly_temp_reservoir_spawning_season=hsi_instance.mean_weekly_temp_reservoir_spawning_season,
            v7a_pct_vegetated=hsi_instance.pct_vegetated,
            v7b_water_depth_spawning_season=hsi_instance.water_depth_spawning_season,
            dem_480=hsi_instance.dem_480,
            hydro_domain_480=hsi_instance.hydro_domain_480,
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
        self.si_7 = self.calculate_si_7()

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _setup_logger(self):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("GizzardShadHSI")
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

    def _create_template_array(self) -> np.ndarray:
        """Create an array from a template all valid pixels are 999.0, and
        NaN from the input are persisted.
        """
        arr = np.where(
            np.isnan(self.v7b_water_depth_spawning_season), np.nan, 999.0
        )
        return arr

    def calculate_si_1(self) -> np.ndarray:
        """LOG10 TOTAL DISSOLVED SOLIDS (PPM) DURING SUMMER GROWING SEASON"""
        self._logger.info("Running SI 1")
        si_1 = self.template.copy()

        # Set to ideal – there is no food limitation
        if self.v1_tds_summer_growing_season is None:
            self._logger.info(
                "TDS during summer growing season data assumes ideal conditions. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else: 
            # condition 1
            mask_1 = (self.v1_tds_summer_growing_season >= 0) & (
                self.v1_tds_summer_growing_season < 1.2
            )
            si_1[mask_1] = 0.0833 * self.v1_tds_summer_growing_season[mask_1]

            # condition 2
            mask_2 = (self.v1_tds_summer_growing_season >= 1.2) & (
                self.v1_tds_summer_growing_season < 3
            )
            si_1[mask_2] = (
                0.5 * (self.v1_tds_summer_growing_season[mask_2])
            ) - 0.5

            # condition 3
            mask_3 = (self.v1_tds_summer_growing_season >= 3) & (
                self.v1_tds_summer_growing_season < 4
            )
            si_1[mask_3] = (
                -1 * (self.v1_tds_summer_growing_season[mask_3])
            ) + 4

            # condition 4
            mask_4 = self.v1_tds_summer_growing_season > 4
            si_1[mask_4] = 0

        if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return si_1

    def calculate_si_2(self) -> np.ndarray:
        """GROWING SEASON (AVERAGE NUMBER OF DAYS BETWEEN LAST SPRING AND FIRST FALL FROST ANUALLY."""
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()

        # Set to Ideal
        if self.v2_avg_num_frost_free_days_growing_season is None:
            self._logger.info(
                "avg num of frost free days in growing season data assumes ideal conditions. Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            # condition 1
            mask_1 = (self.v2_avg_num_frost_free_days_growing_season >= 80) & (
                self.v2_avg_num_frost_free_days_growing_season < 105
            )
            si_2[mask_1] = (
                0.002 * (self.v2_avg_num_frost_free_days_growing_season[mask_1])
            ) - 0.11

            # condition 2
            mask_2 = (self.v2_avg_num_frost_free_days_growing_season >= 105) & (
                self.v2_avg_num_frost_free_days_growing_season < 245
            )
            si_2[mask_2] = (
                0.0061 * (self.v2_avg_num_frost_free_days_growing_season[mask_2])
            ) - 0.5375

            # condition 3
            mask_3 = (self.v2_avg_num_frost_free_days_growing_season >= 245) & (
                self.v2_avg_num_frost_free_days_growing_season <= 265
            )
            si_2[mask_3] = (
                0.0019 * (self.v2_avg_num_frost_free_days_growing_season[mask_3])
            ) + 0.4963

            # condition 4
            mask_4 = self.v2_avg_num_frost_free_days_growing_season > 265
            si_2[mask_4] = 1

        if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")
        
        return si_2

    def calculate_si_3(self) -> np.ndarray:
        """MEAN WEEKLY SUMMER TEMPERATURE (EPILIMNION) (°C)."""
        self._logger.info("Running SI 3")
        # initialize with NaN from depth array, else 999
        si_3 = self.template.copy()

        # Set to ideal 
        if self.v3_mean_weekly_summer_temp is None:
            self._logger.info(
                "mean weekly summer temperature data assumes ideal conditions. Setting index to 1."
            )
            si_3[~np.isnan(si_3)] = 1

        else:
            # condition 1
            mask_1 = (self.v3_mean_weekly_summer_temp >= 15) & (
                self.v3_mean_weekly_summer_temp < 18.5
            )
            si_3[mask_1] = (
                0.0286 * (self.v3_mean_weekly_summer_temp[mask_1])
            ) - 0.4286

            # condition 2
            mask_2 = (self.v3_mean_weekly_summer_temp >= 18.5) & (
                self.v3_mean_weekly_summer_temp < 22
            )
            si_3[mask_2] = (
                0.2571 * (self.v3_mean_weekly_summer_temp[mask_2])
            ) - 4.6571

            # condition 3
            mask_3 = (self.v3_mean_weekly_summer_tempn >= 22) & (
                self.v3_mean_weekly_summer_temp <= 29
            )
            si_3[mask_3] = 1

            # condition 4
            mask_4 = (self.v3_mean_weekly_summer_tempn > 29) & (
                self.v3_mean_weekly_summer_temp < 33
            )
            si_3[mask_4] = (
                -0.1875 * (self.v3_mean_weekly_summer_temp[mask_4])
            ) + 6.4375

            # condition 5
            mask_5 = (self.v3_mean_weekly_summer_temp >= 33) & (
                self.v3_mean_weekly_summer_temp <= 35
            )
            si_3[mask_5] = (
                -0.1 * (self.v3_mean_weekly_summer_temp[mask_5])
            ) + 3.55

        return si_3

    def calculate_si_4(self) -> np.ndarray:
        """MAXIMUM AVAILABLE DISSOLVED OXYGEN IN EPILIMNION DURING SUMMER STRATIFICATION (JUL - SEPT)."""
        self._logger.info("Running SI 4")
        si_4 = self.template.copy()

        if self.v4_max_do_summer is None:
            self._logger.info(
                "Maximum available dissolved oxygen in epilimnion during summer stratification"
                "is not provided. Setting index to 1."
            )
            si_4[~np.isnan(si_4)] = 1

        else:
            # condition 1
            mask_1 = self.v4_max_do_summer <= 1
            si_4[mask_1] = 0

            # condition 2
            mask_2 = (self.v4_max_do_summer > 1) & (
                self.v4_max_do_summer < 6
            )
            si_4[mask_2] = (
                0.2 * (self.v4_max_do_summer[mask_2])
            ) - 0.2

            # condition 3
            mask_3 = self.v4_max_do_summer > 6
            si_4[mask_3] = 1

        if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return si_4

    def calculate_si_5(self) -> np.ndarray:
        """WATER LEVEL DURING SPAWNING SEASON AND EMBRYO DEVELOPMENT."""
        self._logger.info("Running SI 5")
        si_5 = self.template.copy()

        # Set to ideal
        if self.v5a_water_lvl_change is None:
            self._logger.info(
                "Water level during spawning season assumes ideal conditions. Setting index to 1."
            )
            si_5[~np.isnan(si_5)] = 1

        else:
            # condition 1: level = 1, rising water levels (wl) and inundated veg
            mask_1 = (self.v5a_water_lvl_change > 0) & (
                (self.v5b_is_veg_inundated == True)
            )
            si_5[mask_1] = 1

            # condition 2: level = 2, stable wl or no inundated veg
            mask_2 = (self.v5a_water_lvl_change == 0) | (
                (self.v5b_is_veg_inundated == False)
            )
            si_5[mask_2] = 0.8

            # condition 3: level = 3, decline (negative change) in wl <= 0.5m
            mask_3 = (self.v5a_water_lvl_change >= -0.5) & (
                self.v5a_water_lvl_change < 0 
            ) & (self.v5b_is_veg_inundated == True)
            si_5[mask_3] = 0.5

            # condition 4: level = 4, decline (negative change) in wl > 0.5m
            mask_4 = (self.v5a_water_lvl_change < -0.5) & (
                self.v5b_is_veg_inundated == True
            )
            si_5[mask_4] = 0.2

        if np.any(np.isclose(si_5, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")
        
        return si_5

    def calculate_si_6(self) -> np.ndarray:
        """MEAN WEEKLY TEMPERATURE IN TRIBUTARIES OR UPPER END OF LAKE OR RESERVOIR DURING SPAWNING SEASON (APR - JUN)."""
        self._logger.info("Running SI 6")
        si_6 = self.template.copy()

        if self.v6_mean_weekly_temp_reservoir_spawning_season is None:
            self._logger.info(
                "Mean weekly temperature in reservoirs during spawning season data not provided. Setting index to 1."
            )
            si_6[~np.isnan(si_6)] = 1

        else: 
            # condition 1
            mask_1 = self.v6_mean_weekly_temp_reservoir_spawning_season <= 10.9
            si_6[mask_1] = 0

            # condition 2
            mask_2 = (self.v6_mean_weekly_temp_reservoir_spawning_season > 10.9) & (
                self.v6_mean_weekly_temp_reservoir_spawning_season <= 15.8
            )
            si_6[mask_2] = (
                0.2041 * (self.v6_mean_weekly_temp_reservoir_spawning_season[mask_2])
            ) - 2.2245

            # condition 3
            mask_3 =  (self.v6_mean_weekly_temp_reservoir_spawning_season > 15.8) & (
                self.v6_mean_weekly_temp_reservoir_spawning_season <= 22.7
            )
            si_6[mask_3] = 1

            # condition 4
            mask_4 = (self.v6_mean_weekly_temp_reservoir_spawning_season > 22.7) & (
                self.v6_mean_weekly_temp_reservoir_spawning_season <= 25.3
            )
            si_6[mask_4] = (
                -0.1923 * (self.v6_mean_weekly_temp_reservoir_spawning_season[mask_4])
            ) + 5.3654

            # condition 5
            mask_5 = (self.v6_mean_weekly_temp_reservoir_spawning_season > 25.3) & (
                self.v6_mean_weekly_temp_reservoir_spawning_season <= 30
            )
            si_6[mask_5] = (
                -0.1064 * (self.v6_mean_weekly_temp_reservoir_spawning_season[mask_5])
            ) + 3.1915

        if np.any(np.isclose(si_6, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return si_6

    def calculate_si_7(self) -> np.ndarray:
        """% AREA VEGETATED AND ≤ 2m DEEP DURING SPAWNING SEASON (APR - JUN)."""
     
        self._logger.info("Running SI 7")
        si_7 = self.template.copy()

        # use Curve A
        # condition 1
        mask_1 = (self.v7a_pct_vegetated <= 10) & (
            self.v7b_water_depth_spawning_season <= 2
        )
        si_7[mask_1] = 0.08 * self.v7a_pct_vegetated[mask_1]

        # condition 2
        mask_2 = (
            (self.v7a_pct_vegetated > 10)
            & (self.v7a_pct_vegetated <= 15)
            & (self.v7b_water_depth_spawning_season <= 2)
        )
        si_7[mask_2] = (0.04 * self.v7a_pct_vegetated[mask_2]) + 0.4

        # condition 3
        mask_3 = (self.v7a_pct_vegetated > 15) & (
            self.v7b_water_depth_spawning_season <= 2
        )
        si_7[mask_3] = 1

        # condition 4
        mask_4 = self.v7b_water_depth_spawning_season > 2
        si_7[mask_4] = 0

        if np.any(np.isclose(si_7, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return si_7

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Gizzard Shad final HSI.")
        for si_name, si_array in [
            ("SI 1", self.si_1),
            ("SI 2", self.si_2),
            ("SI 3", self.si_3),
            ("SI 4", self.si_4),
            ("SI 5", self.si_5),
            ("SI 6", self.si_6),
            ("SI 7", self.si_7),
        ]:
            invalid_values = (si_array < 0) | (si_array > 1)
            if np.any(invalid_values):
                num_invalid = np.count_nonzero(invalid_values)
                self._logger.warning(
                    "%s contains %d values outside the range [0, 1].",
                    si_name,
                    num_invalid,
                )

        # individual model components
        self.food_component = self.si_1  # will be 1 for hec-ras
        self.water_quality = (
            np.minimum(self.si_3, self.si_4) * self.si_2
        )  
        self.reproduction = (self.si_5 + self.si_6 + self.si_7) / 3

        # combine individual suitability indices
        hsi = np.minimum(
            self.food_component, np.minimum(self.water_quality, self.reproduction)
        )  

        # Note on np.minimum(): If one of the elements being compared is NaN (Not a Number), NaN is returned.
        # Check the final HSI array for invalid values
        invalid_values_hsi = (hsi < 0) | (hsi > 1)
        if np.any(invalid_values_hsi):
            num_invalid_hsi = np.count_nonzero(invalid_values_hsi)
            self._logger.warning(
                "Final HSI contains %d values outside the range [0, 1].",
                num_invalid_hsi,
            )

        # subset final HSI array to vegetation domain (not hydrologic domain)
        # Masking: Set values in `mask` to NaN wherever `data` is NaN
        masked_hsi = np.where(np.isnan(self.dem_480), np.nan, hsi)

        return masked_hsi
