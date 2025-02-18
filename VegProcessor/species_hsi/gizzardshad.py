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

    # gridded data as numpy arrays or None
    # init with None to be distinct from np.nan
    v1_tds_summer_growing_season: np.ndarray = None #ideal
    v2_avg_num_frost_free_days_growing_season: np.ndarray = None #ideal
    v3_mean_weekly_summer_temp: np.ndarray = None #ideal
    v4_max_do_summer: np.ndarray = None #ideal
    v5_water_lvl_spawning_season: np.ndarray = None #ideal
    v6_mean_weekly_temp_reservoir_spawning_season: np.ndarray = None #ideal
    #v7_pct_vegetated_and_2m_depth_spawning_season : np.ndarray = None #use curve A
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
            v5_water_lvl_spawning_season=hsi_instance.water_lvl_spawning_season,
            v6_mean_weekly_temp_reservoir_spawning_season=hsi_instance.mean_weekly_temp_reservoir_spawning_season,
            v7a_pct_vegetated=hsi_instance.pct_vegetated,
            v7b_water_depth_spawning_season=hsi_instance.water_depth_spawning_season,
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
        """LOG10 TOTAL DISSOLVED SOLIDS (PPM) DURING SUMMER GROWING SEASON"""
        # Set to ideal – there is no food limitation
        if self.v1_tds_summer_growing_season is None:
            #self._logger.info("TDS during summer growing season data not provided. Setting index to 1.")
            self._logger.info("TDS during summer growing season data assumes ideal conditions. Setting index to 1.")
            si_1 = np.ones(self._shape)

        # TODO: This will ALWAYS be set to ideal per hsi specs
        # consider include a diff if/else statement to handle ALWAYS IDEAL cases
        #else:
        #    self._logger.info("Running SI 1")
        #    si_1 = np.full(self._shape, 999.0)
        return si_1

    def calculate_si_2(self) -> np.ndarray:
        """GROWING SEASON (AVERAGE NUMBER OF DAYS BETWEEN LAST SPRING AND FIRST FALL FROST ANUALLY."""
        # Set to Ideal
        if self.v2_avg_num_frost_free_days_growing_season is None:
            # self._logger.info(
            #     "avg num of frost free days in growing season data not provided. Setting index to 1."
            # )
            self._logger.info(
                "avg num of frost free days in growing season data assumes ideal conditions. Setting index to 1."
            )
            si_2 = np.ones(self._shape)

        # TODO: This will ALWAYS be set to ideal per hsi specs
        # consider include a diff if/else statement to handle ALWAYS IDEAL cases
        #else:
        #    self._logger.info("Running SI 1")
        #    si_1 = np.full(self._shape, 999.0)

        return si_2

    def calculate_si_3(self) -> np.ndarray:
        """MEAN WEEKLY SUMMER TEMPERATURE (EPILIMNION) (°C)."""
        # Set to ideal
        if self.v3_mean_weekly_summer_temp is None:
            # self._logger.info(
            #     "mean weekly summer temperature data not provided. Setting index to 1."
            # )
            self._logger.info(
                "mean weekly summer temperature data assumes ideal conditions. Setting index to 1."
            )
            si_3 = np.ones(self._shape)

        # TODO: This will ALWAYS be set to ideal per hsi specs
        # consider include a diff if/else statement to handle ALWAYS IDEAL cases
        #else:
        #    self._logger.info("Running SI 1")
        #    si_1 = np.full(self._shape, 999.0)

        return si_3
    
    def calculate_si_4(self) -> np.ndarray:
        """MAXIMUM AVAILABLE DISSOLVED OXYGEN IN EPILIMNION DURING SUMMER STRATIFICATION."""
        # TMP: Set to ideal for HecRas only
        if self.v4_max_do_summer is None:
            # self._logger.info(
            #     "mean weekly summer temperature data not provided. Setting index to 1."
            # )
            self._logger.info(
                "mean weekly summer temperature data assumes ideal conditions. Setting index to 1."
            )
            si_4 = np.ones(self._shape)

        # TODO: this is a quick fix for hec-ras, need to implement the actual logic for other HH models
        # SI4 = 0, when V4 (ppm) ≤ 1
	    # (0.2*V4) - 0.2, when 1 < V4 ≤ 6
	    # 1, when V4 > 6
        #else:
        #    self._logger.info("Running SI 1")
        #    si_1 = np.full(self._shape, 999.0)

        return si_4
    
    def calculate_si_5(self) -> np.ndarray:
        """WATER LEVEL DURING SPAWNING SEASON AND EMBRYO DEVELOPMENT."""
        # Set to ideal
        if self.v5_water_lvl_spawning_season is None:
            # self._logger.info(
            #     "water level during spawning season data not provided. Setting index to 1."
            # )
            self._logger.info(
                "water level during spawning season assumes ideal conditions. Setting index to 1."
            )
            si_5 = np.ones(self._shape)

        # TODO: This will ALWAYS be set to ideal per hsi specs
        # consider include a diff if/else statement to handle ALWAYS IDEAL cases
        #else:
        #    self._logger.info("Running SI 1")
        #    si_1 = np.full(self._shape, 999.0)

        return si_5

    def calculate_si_6(self) -> np.ndarray:
        """MAXIMUM AVAILABLE DISSOLVED OXYGEN IN EPILIMNION DURING SUMMER STRATIFICATION."""
        # TMP: Set to ideal for HecRas only (20 degrees C)
        # April - June is considered spawning season
        if self.v6_mean_weekly_temp_reservoir_spawning_season is None:
            # self._logger.info(
            #     "mean weekly temperature in reservoirs during spawning season data not provided. Setting index to 1."
            # )
            self._logger.info(
                "mean weekly temperature in reservoirs during spawning season data assumes ideal conditions. Setting index to 1."
            )
            si_6 = np.ones(self._shape)

        # TODO: this is a quick fix for hec-ras, need to implement the actual logic for other HH models
        # SI6 = 0, when V6 ≤ 10
        # (0.2041*V6) – 2.2245, when 10.9 < V6 ≤ 15.8
        # 1, when 15.8 < V6 ≤ 22.7
        # (-0.1923*V6) + 5.3654, when 22.7 < V6 ≤ 25.3
        # (-0.1064*V6) + 3.1915, when 25.3 < V6 ≤ 30.2
        #else:
        #    self._logger.info("Running SI 1")
        #    si_1 = np.full(self._shape, 999.0)

        return si_6

    def calculate_si_7(self) -> np.ndarray:
        """% AREA VEGETATED AND ≤ 2m DEEP DURING SPAWNING SEASON."""
        # Use Curve A - Spawning season April-June in Upper Barataria
        self._logger.info("Running SI 7")
        
        # for array in [
        #     self.v7a_pct_vegetated,
        #     self.v7b_water_depth_spawning_season,
        # ]:
        #     if array is None:
        #         self._logger.info("pct vegetated or water depth during spawning season data not provided. Setting index to 1.", array)
        #         array = np.ones(self._shape)
        #         #si_7 = np.ones(self._shape)

        # Create an array to store the results
        si_7 = np.full(self._shape, 0.0) #should thid be 999?
        
        # calc pct first, shorthand, yey.
        self.v7a_pct_vegetated /= 100
        
        # condition 1
        mask_1 = (self.v7a_pct_vegetated <= 10) & (self.v7b_water_depth_spawning_season <= 2)
        si_7[mask_1] = (0.08 * self.v7a_pct_vegetated[mask_1])

        # condition 2
        mask_2 = (self.v7a_pct_vegetated > 10) & (self.v7a_pct_vegetated <= 15) & (self.v7b_water_depth_spawning_season <= 2)
        si_7[mask_2] = (0.04 * self.v7a_pct_vegetated[mask_2]) + 0.4

        # condition 3 USE CURVE A
        mask_3 = (self.v7a_pct_vegetated > 15) & (self.v7b_water_depth_spawning_season <= 2) #& (self.v7_pct_vegetated_and_2m_depth_spawning_season <= 30)
        si_7[mask_3] = 1

        mask_4 = (self.v7b_water_depth_spawning_season > 2)
        si_7[mask_4] = 0
        
        # Create an array to store the results
        #si_7 = np.full(self._shape, 999.0)
        
        # if self.v7b_water_depth_spawning_season.any() > 2.0:
        #     si_7 = np.zeros(self._shape) #is this right?
        # else:
        #     #calc pct first, shorthand, yey. 
        #     self.v7a_pct_vegetated /= 100
        #     # condition 1
        #     mask_1 = self.v7a_pct_vegetated <= 10
        #     si_7[mask_1] = (0.08 * self.v7a_pct_vegetated[mask_1])

        #     # condition 2
        #     mask_2 = (self.v7a_pct_vegetated > 10) & (self.v7a_pct_vegetated <= 15)
        #     si_7[mask_2] = (0.04 * self.v7a_pct_vegetated[mask_2]) + 0.4

        #     # condition 3 USE CURVE A
        #     mask_3 = self.v7a_pct_vegetated > 15 #& (self.v7_pct_vegetated_and_2m_depth_spawning_season <= 30)
        #     si_7[mask_3] = 1

        # Check for unhandled condition with tolerance
        # if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
        #     raise ValueError("Unhandled condition in SI logic!")

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

        # Set up components and equations
        food_component: np.ndarray = field(init=False)
        water_quality: np.ndarray = field(init=False)
        reproduction: np.ndarray = field(init=False)

        # TODO: may want to move these outside calculate_overall_suitability() into their own methods
        # so they can be accessed individually
        food_component = self.si_1 # will be 1 for hec-ras
        water_quality = np.minimum(self.si_3, self.si_4) * self.si_2 # will be 1 for hec-ras
        reproduction = (self.si_5 + self.si_6 + self.si_7) / 3
        
        #hsi = reproduction
        hsi = np.minimum(food_component, np.minimum(water_quality, reproduction)) # will be reproduction for hec-ras

        # Note on np.minimum(): If one of the elements being compared is NaN (Not a Number), NaN is returned.

        # Check the final HSI array for invalid values
        invalid_values_hsi = (hsi < 0) | (hsi > 1)
        if np.any(invalid_values_hsi):
            num_invalid_hsi = np.count_nonzero(invalid_values_hsi)
            self._logger.warning(
                "Final HSI contains %d values outside the range [0, 1].",
                num_invalid_hsi,
            )

        return hsi
