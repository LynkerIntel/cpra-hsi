from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class RiverineCatfishHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    hydro_domain_480: np.ndarray = None
    dem_480: np.ndarray = None

    v1_pct_pools_avg_summer_flow: np.ndarray = None
    v2_pct_cover_in_summer_pools_bw: np.ndarray = None
    v4_fpp_substrate_avg_summer_flow: np.ndarray = None
    v5_avg_temp_in_midsummer_pools_bw: np.ndarray = None
    v6_grow_season_length_frost_free_days: np.ndarray = None
    v7_max_monthly_avg_summer_turbidity: np.ndarray = None
    v8_avg_min_do_in_midsummer_pools_bw: np.ndarray = None
    v9_max_summer_salinity: np.ndarray = None
    v10_avg_temp_in_spawning_embryo_pools_bw: np.ndarray = None
    v11_max_salinity_spawning_embryo: np.ndarray = None
    v12_avg_midsummer_temp_in_pools_bw_fry: np.ndarray = None
    v13_max_summer_salinity_fry_juvenile: np.ndarray = None
    v14_avg_midsummer_temp_in_pools_bw_juvenile: np.ndarray = None
    v18_avg_vel_summer_flow: np.ndarray = None

   

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)
    si_3: np.ndarray = field(init=False)
    si_4: np.ndarray = field(init=False)
    si_5: np.ndarray = field(init=False)
    si_6: np.ndarray = field(init=False)
    si_7: np.ndarray = field(init=False)
    si_8: np.ndarray = field(init=False)
    si_9: np.ndarray = field(init=False)
    si_10: np.ndarray = field(init=False)
    si_11: np.ndarray = field(init=False)
    si_12: np.ndarray = field(init=False)
    si_13: np.ndarray = field(init=False)
    si_14: np.ndarray = field(init=False)
    si_18: np.ndarray = field(init=False)


    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create Riverine Catfish HSI instance from an HSI instance."""

        return cls(
            v1_pct_pools_avg_summer_flow=hsi_instance.pct_pct_pools_avg_summer_flow,
            v2_pct_cover_in_summer_pools_bw=hsi_instance.pct_cover_in_summer_pools_bw, #set to ideal
            v4_fpp_substrate_avg_summer_flow=hsi_instance.fpp_substrate_avg_summer_flow,#set to ideal
            v5_avg_temp_in_midsummer_pools_bw=hsi_instance.avg_temp_in_midsummer_pools_bw,
            v6_grow_season_length_frost_free_days=hsi_instance.pct_grow_season_length_frost_free_days,
            v7_max_monthly_avg_summer_turbidity=hsi_instance.pct_max_monthly_avg_summer_turbidity,
            v8_avg_min_do_in_midsummer_pools_bw=hsi_instance.avg_min_do_in_midsummer_pools_bw,
            v9_max_summer_salinity=hsi_instance.max_summer_salinity, 
            v10_avg_temp_in_spawning_embryo_pools_bw=hsi_instance.avg_temp_in_spawning_embryo_pools_bw,
            v11_max_salinity_spawning_embryo=hsi_instance.max_salinity_spawning_embryo, 
            v12_avg_midsummer_temp_in_pools_bw_fry=hsi_instance.avg_midsummer_temp_in_pools_bw_fry,
            v13_max_summer_salinity_fry_juvenile=hsi_instance.max_summer_salinity_fry_juvenile, 
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
        self.si_4 = self.calculate_si_4()
        self.si_5 = self.calculate_si_5()
        self.si_6 = self.calculate_si_6()
        self.si_7 = self.calculate_si_7()
        self.si_8 = self.calculate_si_8()
        self.si_9 = self.calculate_si_9()
        self.si_10 = self.calculate_si_10()
        self.si_11 = self.calculate_si_11()
        self.si_12 = self.calculate_si_12()
        self.si_13 = self.calculate_si_13()
        self.si_14 = self.calculate_si_14()
        self.si_18 = self.calculate_si_18()

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _create_template_array(self) -> np.ndarray:
        """Create an array from a template all valid pixels are 999.0, and
        NaN from the input are persisted.
        """
        # Riverine Catfish has depth related vars, and is
        # limited to hydrologic model domain
        arr = np.where(np.isnan(self.hydro_domain_480), np.nan, 999.0)
        return arr

    def clip_array(self, result: np.ndarray) -> np.ndarray:
        """Clip array values to between 0 and 1, for cases
        where equations may result in slightly higher than 1.
        Only logs a warning if the input array has values greater than 1.1,
        for cases where there may be a logical error.
        """
        clipped = np.clip(result, 0.0, 1.0)
        if np.any(result > 1.1):
            self._logger.warning(
                "SI output clipped to [0, 1]. SI arr includes values > 1.1, check logic!"
            )
        return clipped

    def _setup_logger(self):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("RiverineCatfishHSI")
        self._logger.setLevel(logging.INFO)

        # Prevent adding multiple handlers if already added
        if not self._logger.handlers:
            # Create console handler and set level
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Create formatter and add it to the handler
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)

            # Add the handler to the logger
            self._logger.addHandler(ch)

    def calculate_si_1(self) -> np.ndarray:
        """Percent pools during average summer flow"""
        self._logger.info("Running SI 1")
        si_1 = self.template.copy()

        if self.v1_pct_pools_avg_summer_flow is None:
            self._logger.info(
                "Percent pools during avg summer flow is not provided. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else: 
            # class 1
            mask_1 = (self.v1_pct_pools_avg_summer_flow < 40)
            si_1[mask_1] = (
                0.02 * (self.v1_pct_pools_avg_summer_flow [mask_1])
            ) + 0.2 
            
            # class 2
            mask_2 = (
                (self.v1_pct_pools_avg_summer_flow >= 40) & 
                (self.v1_pct_pools_avg_summer_flow <= 60)
            )
            si_1[mask_2] = 1

            # class 3
            mask_3 = (
                (self.v1_pct_pools_avg_summer_flow > 60) & 
                (self.v1_pct_pools_avg_summer_flow <= 100)
            )
            si_1[mask_3] = (
                -0.0125 * (self.v1_pct_pools_avg_summer_flow [mask_1])
            ) + 1.75 

        if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_1)

    def calculate_si_2(self) -> np.ndarray:
        """Percent cover (logs, boulders, cavities, brush, debris, or standing timber) 
        during summer within pools, backwater areas """
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()

        if self.v2_pct_cover_in_summer_pools_bw is None:
            self._logger.info(
                "Pct cover during summer within pools, backwaters is not provided. ""Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            raise NotImplementedError(
                "No logic for catfish v2 exists. Either use ideal (set array None) or add logic."
            )
    
        if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")
        
        return self.clip_array(si_2)

    def calculate_si_4(self) -> np.ndarray:
        """Food production potential in river by substrate type present during average summer flow"""
        self._logger.info("Running SI 4")
        si_4 = self.template.copy()

        if self.v4_fpp_substrate_avg_summer_flow is None:
            self._logger.info(
                "Food production potential data is not provided. Setting index to 1."
            )
            si_4[~np.isnan(si_4)] = 1

        else:
            raise NotImplementedError(
                "No logic for catfish v4 exists. Either use ideal (set array None) or add logic."
            )

        if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_4)

    def calculate_si_5(self) -> np.ndarray:
        """Size of Contiguous Forested Area in Acres"""
        self._logger.info("Running SI 5")
        si_5 = self.template.copy()

        if self.v5_size_forested_area is None:
            self._logger.info(
                "Size of contiguous forested area in acres not provided. Setting index to 1."
            )
            si_5[~np.isnan(si_5)] = 1

        else: 

            # condition 1 for class 1
            mask_1 = (self.v5_size_forested_area >= 0) & (
                self.v5_size_forested_area <= 5
            )
            si_5[mask_1] = 0.2

            # condition 2 for class 2
            mask_2 = (self.v5_size_forested_area > 5) & (
                self.v5_size_forested_area <= 20
            )
            si_5[mask_2] = 0.4

            # condition 3 for class 3
            mask_3 = (self.v5_size_forested_area > 20) & (
                self.v5_size_forested_area <= 100
            )
            si_5[mask_3] = 0.6

            # condition 4 for class 4
            mask_4 = (self.v5_size_forested_area > 100) & (
                self.v5_size_forested_area <= 500
            )
            si_5[mask_4] = 0.8

            # condition 5 for class 5
            mask_5 = (self.v5_size_forested_area > 500)
            si_5[mask_5] = 1

        si_5 = self.blh_cover_mask(si_5)

        if np.any(np.isclose(si_5, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")
        
        return self.clip_array(si_5)

    def calculate_si_6(self) -> np.ndarray:
        """Suitability and Traversability of Surrounding Land Uses"""
        self._logger.info("Running SI 6")
        si_6 = self.template.copy()

        # Set to ideal
        if self.v6_suit_trav_surr_lu is None:
            self._logger.info(
                "Suit and Trav of Surrounding Land Uses assumes ideal conditions. Setting index to 1."
            )
            si_6[~np.isnan(si_6)] = 1

        else:
            raise NotImplementedError(
                "No logic for bottomland hardwood v6 exists. Either use ideal (set array None) or add logic."
            )

        si_6 = self.blh_cover_mask(si_6)

        if np.any(np.isclose(si_6, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_6)

    def calculate_si_7(self) -> np.ndarray:
        """Disturbance"""
        self._logger.info("Running SI 7")
        si_7 = self.template.copy()

        # Set to ideal.
        if self.v7_disturbance is None:
            self._logger.info(
                "Disturbance assumes ideal conditions. Setting index to 1."
            )
            si_7[~np.isnan(si_7)] = 1

        else:
            raise NotImplementedError(
                "No logic for bottomland hardwood v7 exists. Either use ideal (set array None) or add logic."
            )

        si_7 = self.blh_cover_mask(si_7)

        if np.any(np.isclose(si_7, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_7)


    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Bottomland Hardwood final HSI.")
        hsi = self.template.copy()
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

        # Combine individual suitability indices
        # condition 1 (tree age < 7)
        mask_1 = self.v2_stand_maturity < 7
        hsi[mask_1] = ((self.si_2[mask_1] ** 4) * 
                       (self.si_4[mask_1] ** 2)
        ) ** (1 / 6)
    
        # condition 2 (tree age >= 7 and v3_understory/midstory data is available)
        mask_2 = self.v2_stand_maturity >= 7
        hsi[mask_2] = ((self.si_1[mask_2] ** 4) * 
                       (self.si_2[mask_2] ** 4) *
                       (self.si_3[mask_2] ** 2) *
                       (self.si_4[mask_2] ** 2) *
                       (self.si_5[mask_2])
        ) ** (1 / 13)

        # Quality control check for invalid values: Ensure combined_score is between 0 and 1
        invalid_values = (hsi < 0) | (hsi > 1)
        if np.any(invalid_values):
            num_invalid = np.count_nonzero(invalid_values)
            self._logger.warning(
                "Combined suitability score has %d values outside [0,1]",
                num_invalid,
            )

        # subset final HSI array to vegetation domain (not hydrologic domain)
        # Masking: Set values in `mask` to NaN wherever `data` is NaN
        masked_hsi = np.where(np.isnan(self.dem_480), np.nan, hsi)

        return masked_hsi
