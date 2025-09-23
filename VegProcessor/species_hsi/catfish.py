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
    si_15: np.ndarray = field(init=False)
    si_16: np.ndarray = field(init=False)
    si_17: np.ndarray = field(init=False)
    si_18: np.ndarray = field(init=False)

    # components and equations
    fc: np.ndarray = field(init=False)
    cc: np.ndarray = field(init=False)
    wq: np.ndarray = field(init=False)
    rc: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    initial_hsi: np.ndarray = field(init=False)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create Riverine Catfish HSI instance from an HSI instance."""

        return cls(
            v1_pct_pools_avg_summer_flow=hsi_instance.pct_pools_avg_summer_flow,
            v2_pct_cover_in_summer_pools_bw=hsi_instance.pct_cover_in_summer_pools_bw, # set to ideal
            v4_fpp_substrate_avg_summer_flow=hsi_instance.fpp_substrate_avg_summer_flow,# set to ideal
            v5_avg_temp_in_midsummer_pools_bw=hsi_instance.avg_temp_in_midsummer_pools_bw,
            v6_grow_season_length_frost_free_days=hsi_instance.grow_season_length_frost_free_days, # if using, set to ideal
            v7_max_monthly_avg_summer_turbidity=hsi_instance.max_monthly_avg_summer_turbidity,
            v8_avg_min_do_in_midsummer_pools_bw=hsi_instance.avg_min_do_in_midsummer_pools_bw,
            v9_max_summer_salinity=hsi_instance.max_summer_salinity, 
            v10_avg_temp_in_spawning_embryo_pools_bw=hsi_instance.avg_temp_in_spawning_embryo_pools_bw,
            v11_max_salinity_spawning_embryo=hsi_instance.max_salinity_spawning_embryo, 
            v12_avg_midsummer_temp_in_pools_bw_fry=hsi_instance.avg_midsummer_temp_in_pools_bw_fry,
            v13_max_summer_salinity_fry_juvenile=hsi_instance.max_summer_salinity_fry_juvenile,
            v14_avg_midsummer_temp_in_pools_bw_juvenile=hsi_instance.avg_midsummer_temp_in_pools_bw_juvenile,
            v18_avg_vel_summer_flow=hsi_instance.avg_vel_summer_flow,
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
                "Percent pools during avg summer flow data is not provided. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else: 
            # condition 1
            mask_1 = (self.v1_pct_pools_avg_summer_flow < 40)
            si_1[mask_1] = (
                0.02 * (self.v1_pct_pools_avg_summer_flow[mask_1])
            ) + 0.2 
            
            # condition 2
            mask_2 = (
                (self.v1_pct_pools_avg_summer_flow >= 40) & 
                (self.v1_pct_pools_avg_summer_flow <= 60)
            )
            si_1[mask_2] = 1

            # condition 3
            mask_3 = (
                (self.v1_pct_pools_avg_summer_flow > 60) & 
                (self.v1_pct_pools_avg_summer_flow <= 100)
            )
            si_1[mask_3] = (
                -0.0125 * (self.v1_pct_pools_avg_summer_flow[mask_3])
            ) + 1.75 

        if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_1)

    def calculate_si_2(self) -> np.ndarray:
        """Percent cover (logs, boulders, cavities, brush, debris, or standing timber) 
        during summer within pools, backwater areas """
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()

        # Set to ideal
        if self.v2_pct_cover_in_summer_pools_bw is None:
            self._logger.info(
                "Pct cover during summer within pools, backwaters assumes ideal conditions. " \
                "Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            # condition 1
            mask_1 = (
                (self.v2_pct_cover_in_summer_pools_bw >= 0) & 
                (self.v2_pct_cover_in_summer_pools_bw < 40)
            )
            si_2[mask_1] = (
                0.022 * (self.v2_pct_cover_in_summer_pools_bw[mask_1])
            ) + 0.1397
            
            # condition 2
            mask_2 = (
                (self.v2_pct_cover_in_summer_pools_bw >= 40)
            )
            si_2[mask_2] = 1
    
        if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")
        
        return self.clip_array(si_2)
    
    def calculate_si_3(self) -> np.ndarray:
        """No logic exists for si_3.""" 
        return NotImplementedError  

    def calculate_si_4(self) -> np.ndarray:
        """Food production potential in river by substrate type present during average summer flow"""
        self._logger.info("Running SI 4")
        si_4 = self.template.copy()

        # Set to ideal
        if self.v4_fpp_substrate_avg_summer_flow is None:
            self._logger.info(
                "Food production potential data assumes ideal conditions. Setting index to 1."
            )
            si_4[~np.isnan(si_4)] = 1

        else:
            # condition 1 (class A == 1)
            mask_1 = self.v4_fpp_substrate_avg_summer_flow == 1
            si_4[mask_1] = 1

            # condition 2 (class B == 2)
            mask_2 = self.v4_fpp_substrate_avg_summer_flow == 2
            si_4[mask_2] = 0.7

            # condition 3 (class C == 3)
            mask_3 = self.v4_fpp_substrate_avg_summer_flow == 3
            si_4[mask_3] = 0.5

            # condition 4 (class D == 4)
            mask_4 = self.v4_fpp_substrate_avg_summer_flow == 4
            si_4[mask_4] = 0.2

        if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_4)

    def calculate_si_5(self) -> np.ndarray:
        """Average midsummer (July to August) water temperature within pools, backwaters (Adult)"""
        self._logger.info("Running SI 5")
        si_5 = self.template.copy()

        if self.v5_avg_temp_in_midsummer_pools_bw is None:
            self._logger.info(
                "Average midsummer water temperature within pools, backwaters is not provided. Setting index to 1."
            )
            si_5[~np.isnan(si_5)] = 1

        else: 

            # condition 1 
            mask_1 = (self.v5_avg_temp_in_midsummer_pools_bw < 17)
            si_5[mask_1] = 0

            # condition 2
            mask_2 = (self.v5_avg_temp_in_midsummer_pools_bw >= 17) & (
                self.v5_avg_temp_in_midsummer_pools_bw <= 26
            )
            si_5[mask_2] = (
                0.111 * (self.v5_avg_temp_in_midsummer_pools_bw[mask_2])
            ) - 1.8889 

            # condition 3
            mask_3 = (self.v5_avg_temp_in_midsummer_pools_bw > 26) & (
                self.v5_avg_temp_in_midsummer_pools_bw <= 29
            )
            si_5[mask_3] = 1.0

            # condition 4
            mask_4 = (self.v5_avg_temp_in_midsummer_pools_bw > 29) & (
                self.v5_avg_temp_in_midsummer_pools_bw <= 34
            )
            si_5[mask_4] = (
                -0.2 * (self.v5_avg_temp_in_midsummer_pools_bw[mask_4])
            ) + 6.8

            # condition 5
            mask_5 = (self.v5_avg_temp_in_midsummer_pools_bw > 34)
            si_5[mask_5] = 0

        if np.any(np.isclose(si_5, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")
        
        return self.clip_array(si_5)

    def calculate_si_6(self) -> np.ndarray:
        """Length of agricultural growing season (frost- free days)"""
        self._logger.info("Running SI 6")
        si_6 = self.template.copy()

        # If using, set to ideal
        if self.v6_grow_season_length_frost_free_days is None:
            self._logger.info(
                "Length of agricultural growing season is not provided. Setting index to 1."
            )
            si_6[~np.isnan(si_6)] = 1

        else:
            # condition 1 
            mask_1 = (self.v6_grow_season_length_frost_free_days < 17)
            si_6[mask_1] = 0

            # condition 2
            mask_2 = (self.v6_grow_season_length_frost_free_days >= 17) & (
                self.v6_grow_season_length_frost_free_days < 200
            )
            si_6[mask_2] = (
                (2.485904e-05 * (self.v6_grow_season_length_frost_free_days[mask_2]) ** 2)
                + (3.943156e-05 * (self.v6_grow_season_length_frost_free_days[mask_2]) ** 3)
                - 0.006587644
            )

            # condition 3
            mask_3 = (self.v6_grow_season_length_frost_free_days >= 200)
            si_6[mask_3] = 1.0

        if np.any(np.isclose(si_6, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_6)

    def calculate_si_7(self) -> np.ndarray:
        """Maximum monthly average turbidity during summer"""
        self._logger.info("Running SI 7")
        si_7 = self.template.copy()

        if self.v7_max_monthly_avg_summer_turbidity is None:
            self._logger.info(
                "Maximum monthly average turbidity during summer" \
                "is not provided. Setting index to 1."
            )
            si_7[~np.isnan(si_7)] = 1

        else:
            # condition 1 
            mask_1 = (self.v7_max_monthly_avg_summer_turbidity <= 110)
            si_7[mask_1] = 1

            # condition 2
            mask_2 = (self.v7_max_monthly_avg_summer_turbidity > 110) & (
                self.v7_max_monthly_avg_summer_turbidity < 285
            )
            si_7[mask_2] = (
                -0.0046 * (self.v7_max_monthly_avg_summer_turbidity[mask_2])
                + 1.5029
            )

            # condition 3
            mask_3 = (self.v7_max_monthly_avg_summer_turbidity >= 285)
            si_7[mask_3] = 0.2
        
        if np.any(np.isclose(si_7, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_7)
    
    def calculate_si_8(self) -> np.ndarray:
        """Average minimum dissolved oxygen levels within pools, backwaters, during midsummer"""
        self._logger.info("Running SI 8")
        si_8 = self.template.copy()

        if self.v8_avg_min_do_in_midsummer_pools_bw is None:
            self._logger.info(
                "Avg min DO levels within pools, backwaters, during midsummer"
                "is not provided. Setting index to 1."
            )
            si_8[~np.isnan(si_8)] = 1

        else:
            # condition 1 
            mask_1 = (self.v8_avg_min_do_in_midsummer_pools_bw < 1)
            si_8[mask_1] = 0

            # condition 2
            mask_2 = (self.v8_avg_min_do_in_midsummer_pools_bw >= 1) & (
                self.v8_avg_min_do_in_midsummer_pools_bw <= 7
            )
            si_8[mask_2] = (
                0.1667 * (self.v8_avg_min_do_in_midsummer_pools_bw[mask_2])
                - 0.1667
            )

            # condition 3
            mask_3 = (self.v8_avg_min_do_in_midsummer_pools_bw > 7)
            si_8[mask_3] = 1
        
        if np.any(np.isclose(si_8, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_8)

    def calculate_si_9(self) -> np.ndarray:
        """Maximum salinity during summer (Adult)"""
        self._logger.info("Running SI 9")
        si_9 = self.template.copy()

        if self.v9_max_summer_salinity is None:
            self._logger.info(
                "Maximum salinity during summer (Adult) is not provided. Setting index to 1."
            )
            si_9[~np.isnan(si_9)] = 1

        else:
            # condition 1 
            mask_1 = (self.v9_max_summer_salinity > 0) & (
                self.v9_max_summer_salinity <= 1.7
            )
            si_9[mask_1] = 1

            # condition 2
            mask_2 = (self.v9_max_summer_salinity > 1.7) & (
                self.v9_max_summer_salinity < 11.4
            )
            si_9[mask_2] = (
                (0.001235 * (self.v9_max_summer_salinity[mask_2]) ** 3)
                - (0.02587 * (self.v9_max_summer_salinity[mask_2]) ** 2)
                + (0.05215 * (self.v9_max_summer_salinity[mask_2])) 
                + 0.9714
            )

            # condition 3
            mask_3 = (self.v9_max_summer_salinity >= 11.4)
            si_9[mask_3] = 0
        
        if np.any(np.isclose(si_9, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_9)
    
    def calculate_si_10(self) -> np.ndarray:
        """Average water temperatures (May to July) within pools, backwaters, 
        during spawning and embryo development (Embryo)"""
        self._logger.info("Running SI 10")
        si_10 = self.template.copy()

        if self.v10_avg_temp_in_spawning_embryo_pools_bw is None:
            self._logger.info(
                "Avg water temp within pools, backwaters, during spawning and embryo development (Embryo)"
                "is not provided. Setting index to 1."
            )
            si_10[~np.isnan(si_10)] = 1

        else:
            # condition 1 
            mask_1 = (self.v10_avg_temp_in_spawning_embryo_pools_bw <= 15.5)
            si_10[mask_1] = 0

            # condition 2
            mask_2 = (self.v10_avg_temp_in_spawning_embryo_pools_bw > 15.5) & (
                self.v10_avg_temp_in_spawning_embryo_pools_bw <= 26
            )
            si_10[mask_2] = (
                (0.0928 * (self.v10_avg_temp_in_spawning_embryo_pools_bw[mask_2]))
                - 1.4456
            )

            # condition 3
            mask_3 = (self.v10_avg_temp_in_spawning_embryo_pools_bw > 26) & (
                self.v10_avg_temp_in_spawning_embryo_pools_bw <= 27.5
            )
            si_10[mask_3] = 1

            # condition 4
            mask_4 = (self.v10_avg_temp_in_spawning_embryo_pools_bw > 27.5) & (
                self.v10_avg_temp_in_spawning_embryo_pools_bw <= 29.2
            )
            si_10[mask_4] = (
                (-0.5882 * (self.v10_avg_temp_in_spawning_embryo_pools_bw[mask_4]))
                + 17.176
            )

            # condition 5
            mask_5 = (self.v10_avg_temp_in_spawning_embryo_pools_bw > 29.2)
            si_10[mask_5] = 0
        
        if np.any(np.isclose(si_10, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_10)

    def calculate_si_11(self) -> np.ndarray:
        """Maximum salinity during spawning and embryo development (Embryo)"""
        self._logger.info("Running SI 11")
        si_11 = self.template.copy()

        if self.v11_max_salinity_spawning_embryo is None:
            self._logger.info(
                "Maximum salinity during spawning and embryo development (Embryo) is not provided."
                "Setting index to 1."
            )
            si_11[~np.isnan(si_11)] = 1

        else:
            # condition 1 
            mask_1 = (self.v11_max_salinity_spawning_embryo >= 0) & (
                self.v11_max_salinity_spawning_embryo <= 2
            )
            si_11[mask_1] = 1

            # condition 2
            mask_2 = (self.v11_max_salinity_spawning_embryo > 2) & (
                self.v11_max_salinity_spawning_embryo <= 16
            )
            si_11[mask_2] = (
                (0.0008024 * (self.v11_max_salinity_spawning_embryo[mask_2]) ** 3)
                - (0.02161 * (self.v11_max_salinity_spawning_embryo[mask_2]) ** 2)
                + (0.08395 * self.v11_max_salinity_spawning_embryo[mask_2])
                + 0.9093
            )

            # condition 3
            mask_3 = (self.v11_max_salinity_spawning_embryo > 16)
            si_11[mask_3] = 0
        
        if np.any(np.isclose(si_11, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_11)
    
    def calculate_si_12(self) -> np.ndarray:
        """Average midsummer water temperature within pools, backwaters (Fry)"""
        self._logger.info("Running SI 12")
        si_12 = self.template.copy()

        if self.v12_avg_midsummer_temp_in_pools_bw_fry is None:
            self._logger.info(
                "Average midsummer water temperature within pools, backwaters (Fry)" \
                "is not provided. Setting index to 1."
            )
            si_12[~np.isnan(si_12)] = 1

        else:
            # condition 1 
            mask_1 = self.v12_avg_midsummer_temp_in_pools_bw_fry <= 15
            si_12[mask_1] = 0

            # condition 2
            mask_2 = (self.v12_avg_midsummer_temp_in_pools_bw_fry > 15) & (
                self.v12_avg_midsummer_temp_in_pools_bw_fry < 28
            )
            si_12[mask_2] = (
                0.0765 * (self.v12_avg_midsummer_temp_in_pools_bw_fry[mask_2])
                - 1.1892
            )

            # condition 3
            mask_3 = (self.v12_avg_midsummer_temp_in_pools_bw_fry >= 28) & (
                self.v12_avg_midsummer_temp_in_pools_bw_fry <= 30
            )
            si_12[mask_3] = 1

            # condition 4
            mask_4 = (self.v12_avg_midsummer_temp_in_pools_bw_fry > 30) & (
                self.v12_avg_midsummer_temp_in_pools_bw_fry < 36
            )
            si_12[mask_4] = (
                -0.1667 * self.v12_avg_midsummer_temp_in_pools_bw_fry[mask_4]
                + 6
            )

            #condition 5
            mask_5 = self.v12_avg_midsummer_temp_in_pools_bw_fry >= 36
            si_12[mask_5] = 0
        
        if np.any(np.isclose(si_12, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_12)
    
    def calculate_si_13(self) -> np.ndarray:
        """Maximum salinity during summer (Fry, Juvenile)"""
        self._logger.info("Running SI 13")
        si_13 = self.template.copy()

        if self.v13_max_summer_salinity_fry_juvenile is None:
            self._logger.info(
                "Maximum salinity during summer (Fry, Juvenile) is not provided. Setting index to 1."
            )
            si_13[~np.isnan(si_13)] = 1

        else:
            # condition 1 
            mask_1 = (self.v13_max_summer_salinity_fry_juvenile >= 0) & (
                self.v13_max_summer_salinity_fry_juvenile <= 5
            )
            si_13[mask_1] = 1

            # condition 2
            mask_2 = (self.v13_max_summer_salinity_fry_juvenile >= 5) & (
                self.v13_max_summer_salinity_fry_juvenile <= 10
            )
            si_13[mask_2] = (
                -0.2 * (self.v13_max_summer_salinity_fry_juvenile[mask_2])
                + 2
            )

            # condition 3
            mask_3 = self.v13_max_summer_salinity_fry_juvenile > 10
            si_13[mask_3] = 0
        
        if np.any(np.isclose(si_13, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_13)
    
    def calculate_si_14(self) -> np.ndarray:
        """Average midsummer water temperature within pools, backwaters (Juvenile)"""
        self._logger.info("Running SI 14")
        si_14 = self.template.copy()

        if self.v14_avg_midsummer_temp_in_pools_bw_juvenile is None:
            self._logger.info(
                "Average midsummer water temperature within " \
                "pools, backwaters (Juvenile) is not provided. Setting index to 1."
            )
            si_14[~np.isnan(si_14)] = 1

        else:
            # condition 1 
            mask_1 = (self.v14_avg_midsummer_temp_in_pools_bw_juvenile > 10) & (
                self.v14_avg_midsummer_temp_in_pools_bw_juvenile <= 15
            )
            si_14[mask_1] = 0

            # condition 2
            mask_2 = (self.v14_avg_midsummer_temp_in_pools_bw_juvenile > 15) & (
                self.v14_avg_midsummer_temp_in_pools_bw_juvenile < 28
            )
            si_14[mask_2] = (
                0.0765 * (self.v14_avg_midsummer_temp_in_pools_bw_juvenile[mask_2])
                - 1.1892
            )

            # condition 3
            mask_3 = (self.v14_avg_midsummer_temp_in_pools_bw_juvenile >= 28) & (
                self.v14_avg_midsummer_temp_in_pools_bw_juvenile <= 30
            )
            si_14[mask_3] = 1

            #condition 4
            mask_4 = (self.v14_avg_midsummer_temp_in_pools_bw_juvenile > 30) & (
                self.v14_avg_midsummer_temp_in_pools_bw_juvenile < 36
            )
            si_14[mask_4] = (
                -0.1538 * (self.v14_avg_midsummer_temp_in_pools_bw_juvenile[mask_4])
                + 5.6154 
            )

            #condition 5
            mask_5 = self.v14_avg_midsummer_temp_in_pools_bw_juvenile >= 36.5
            si_14[mask_5] = 0
        
        if np.any(np.isclose(si_14, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_14)
    
    def calculate_si_15(self) -> np.ndarray:
        """No logic exists for si_15.""" 
        return NotImplementedError  
    
    def calculate_si_16(self) -> np.ndarray:
        """No logic exists for si_16.""" 
        return NotImplementedError  
    
    def calculate_si_17(self) -> np.ndarray:
        """No logic exists for si_17.""" 
        return NotImplementedError  
    
    def calculate_si_18(self) -> np.ndarray:
        """Average current velocity in cover areas during average summer flow"""
        self._logger.info("Running SI 18")
        si_18 = self.template.copy()

        if self.v18_avg_vel_summer_flow is None:
            self._logger.info(
                "Average current velocity in cover areas during average summer flow"
                "is not provided. Setting index to 1."
            )
            si_18[~np.isnan(si_18)] = 1

        else:
            # condition 1 
            mask_1 = self.v18_avg_vel_summer_flow <= 15
            si_18[mask_1] = 1

            # condition 2
            mask_2 = (self.v18_avg_vel_summer_flow > 15) & (
                self.v18_avg_vel_summer_flow < 38
            )
            si_18[mask_2] = (
                0.001195 * (self.v18_avg_vel_summer_flow[mask_2]) ** 2
                - 0.1025 * (self.v18_avg_vel_summer_flow[mask_2])
                + 2.278
            )

            # condition 3
            mask_3 = self.v18_avg_vel_summer_flow >= 38
            si_18[mask_3] = 0.1

        if np.any(np.isclose(si_18, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_18)

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Catfish final HSI.")
        hsi = self.template.copy()
        for si_name, si_array in [
            ("SI 1", self.si_1),
            ("SI 2", self.si_2),
            ("SI 4", self.si_4),
            ("SI 5", self.si_5),
            ("SI 6", self.si_6),
            ("SI 7", self.si_7),
            ("SI 8", self.si_8), 
            ("SI 9", self.si_9),
            ("SI 10", self.si_10),
            ("SI 11", self.si_11),
            ("SI 12", self.si_12),
            ("SI 13", self.si_13),
            ("SI 14", self.si_14),
            ("SI 18", self.si_18),
        ]:
            invalid_values = (si_array < 0) | (si_array > 1)
            if np.any(invalid_values):
                num_invalid = np.count_nonzero(invalid_values)
                self._logger.warning(
                    "%s contains %d values outside the range [0, 1].",
                    si_name,
                    num_invalid,
                )
        
        # food component (fc)
        self.fc = (self.si_2 + self.si_4) / 2

        # cover component (cc)
        self.cc = (self.si_1 * self.si_2 * self.si_18) ** (1/3)

        # water quality component (wq)
        if self.v5_avg_temp_in_midsummer_pools_bw is None:
            self._logger.info("Temperature data unavailable. Using SI_6 for WQ calculation.")
            wq_term1 = 2 * self.si_6
        else:
            # The data is available, use the standard WQ equation
            wq_term1 = (2 * (self.si_5 + self.si_12 + self.si_14)) / 3
        self.wq = (
            wq_term1 + 
            self.si_7 + 
            2 * (self.si_8) + 
            self.si_9 + 
            self.si_13
        ) / 7

        # water quality component conditions
        wq_mask = (self.si_5 <= 0.4) | (self.si_12 <= 0.4) | (self.si_14 <= 0.4) | \
            (self.si_8 <= 0.4) | (self.si_9 <= 0.4) | (self.si_13 <= 0.4)
        self.wq = np.where(wq_mask, np.minimum(np.stack(
            [self.si_5, self.si_12, self.si_14, self.si_8, self.si_9, self.si_13, self.wq]
            ), axis=0).min(axis=0), self.wq)
        
        # reproduction component (rc)
        self.rc = (
            (((self.si_1) * (self.si_2 ** (2)) * (self.si_8 ** (2)) * 
               (self.si_10 ** (2)) * (self.si_11)) ** (1/8))
        )
        # reproduction component conditions
        rc_mask = (self.si_8 <= 0.4) | (self.si_10 <= 0.4) | (self.si_11 <= 0.4)
        self.rc = np.where(rc_mask, 
                      np.minimum(np.stack(
                          [self.si_6, self.si_10, self.si_11, self.rc]), axis=0).min(axis=0), 
                          self.rc)

        # Combine individual suitability indices
        initial_hsi = (self.fc * self.cc * (self.wq ** 2) * (self.rc ** 2)) ** (1/6)

        # If wq or rc <= 0.4, select min(wq, rc, initial_hsi)
        mask_hsi = (self.wq <= 0.4) | (self.rc <= 0.4)
        hsi = np.where(mask_hsi, np.minimum.reduce([self.wq, self.rc, initial_hsi]), initial_hsi)

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
