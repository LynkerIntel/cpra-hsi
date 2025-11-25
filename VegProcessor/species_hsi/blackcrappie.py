from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class BlackCrappieHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    hydro_domain_480: np.ndarray = None
    dem_480: np.ndarray = None
    water_depth_midsummer: np.ndarray = None

    v1_max_monthly_avg_summer_turbidity: np.ndarray = None
    v2_pct_cover_in_midsummer_pools_overflow_bw: np.ndarray = None
    v3_stream_gradient: np.ndarray = None
    v4_avg_vel_summer_flow_pools_bw: np.ndarray = None
    v5_pct_pools_bw_avg_spring_summer_flow: np.ndarray = None
    v7_ph_year: np.ndarray = None
    v8_most_suit_temp_in_midsummer_pools_bw_adult: np.ndarray = None
    v9_most_suit_temp_in_midsummer_pools_bw_juvenile: np.ndarray = None
    v10_avg_midsummer_temp_in_pools_bw_fry: np.ndarray = None
    v11_avg_spawning_temp_in_bw_embryo: np.ndarray = None
    v12_min_do_in_midsummer_temp_strata: np.ndarray = None
    v13_min_do_in_spawning_bw: np.ndarray = None
    v14_max_salinity_gs: np.ndarray = None

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

    # components and equations
    fc: np.ndarray = field(init=False)
    wq_tcr: np.ndarray = field(init=False)
    wq_tcr_adj: np.ndarray = field(init=False)
    wq_init: np.ndarray = field(init=False)
    wq: np.ndarray = field(init=False)
    rc: np.ndarray = field(init=False)
    ot: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    initial_hsi: np.ndarray = field(init=False)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create Riverine Black Crappie HSI instance from an HSI instance."""

        return cls(
            v1_max_monthly_avg_summer_turbidity=hsi_instance.blackcrappie_max_monthly_avg_summer_turbidity,
            v2_pct_cover_in_midsummer_pools_overflow_bw=hsi_instance.blackcrappie_pct_cover_in_midsummer_pools_overflow_bw,  # set to ideal
            v3_stream_gradient=hsi_instance.blackcrappie_stream_gradient,  # set to ideal
            v4_avg_vel_summer_flow_pools_bw=hsi_instance.blackcrappie_avg_vel_summer_flow_pools_bw,
            v5_pct_pools_bw_avg_spring_summer_flow=hsi_instance.blackcrappie_pct_pools_bw_avg_spring_summer_flow,
            v7_ph_year=hsi_instance.blackcrappie_ph_year,  # set to ideal
            v8_most_suit_temp_in_midsummer_pools_bw_adult=hsi_instance.water_temperature_july_august_mean,
            v9_most_suit_temp_in_midsummer_pools_bw_juvenile=hsi_instance.water_temperature_july_august_mean,
            v10_avg_midsummer_temp_in_pools_bw_fry=hsi_instance.water_temperature_july_august_mean,
            v11_avg_spawning_temp_in_bw_embryo=hsi_instance.water_temperature_feb_march_mean,
            v12_min_do_in_midsummer_temp_strata=hsi_instance.blackcrappie_min_do_in_midsummer_temp_strata,
            v13_min_do_in_spawning_bw=hsi_instance.blackcrappie_min_do_in_spawning_bw,
            v14_max_salinity_gs=hsi_instance.salinity_max_april_sept,
            dem_480=hsi_instance.dem_480,
            hydro_domain_480=hsi_instance.hydro_domain_480,
            water_depth_midsummer=hsi_instance.water_depth_july_august_mean,
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
        self.si_15 = self.calculate_si_15()

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _create_template_array(self, *input_arrays) -> np.ndarray:
        """Create an array from a template where valid pixels are 999.0, and
        NaN values are propagated from hydro domain and optional input arrays.

        Parameters
        ----------
        *input_arrays : np.ndarray, optional
            One or more input arrays from which NaN values will be propagated

        Returns
        -------
        np.ndarray
            Template array with 999.0 for valid pixels and NaN elsewhere
        """
        # Start with hydro domain mask
        arr = np.where(np.isnan(self.hydro_domain_480), np.nan, 999.0)

        # Propagate NaN from any input arrays
        for input_arr in input_arrays:
            if input_arr is not None:
                arr = np.where(np.isnan(input_arr), np.nan, arr)

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

    def backwaters_mask(self, si_array: np.ndarray) -> np.ndarray:
        """DRAFT
        Ensure that areas outside of backwaters are excluded.

        Also ensures areas outside the hydro domain remain NaN.
        """
        ## Apply hydro domain mask
        # si_array = np.where(np.isnan(self.hydro_domain_480), np.nan, si_array)

        backwaters_mask = (self.water_depth_midsummer <= 0) & (
            self.water_depth_midsummer > 3
        )
        si_array[backwaters_mask] = np.nan
        # return si_array
        return NotImplementedError

    def _setup_logger(self):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("BlackCrappieHSI")
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
        """Maximum monthly average turbidity during summer (July - September)"""
        self._logger.info("Running SI 1")
        si_1 = self.template.copy()

        if self.v1_max_monthly_avg_summer_turbidity is None:
            self._logger.info(
                "Maximum monthly average turbidity during summer is not provided. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else:
            # condition 1
            mask_1 = self.v1_max_monthly_avg_summer_turbidity <= 50
            si_1[mask_1] = 1

            # condition 2
            mask_2 = (self.v1_max_monthly_avg_summer_turbidity >= 50) & (
                self.v1_max_monthly_avg_summer_turbidity < 150
            )
            si_1[mask_2] = (
                -0.008 * (self.v1_max_monthly_avg_summer_turbidity[mask_2])
            ) + 1.4

            # condition 3
            mask_3 = (self.v1_max_monthly_avg_summer_turbidity >= 150) & (
                self.v1_max_monthly_avg_summer_turbidity < 190
            )
            si_1[mask_3] = (
                -0.0005 * (self.v1_max_monthly_avg_summer_turbidity[mask_3])
            ) + 0.95

            # condition 4
            mask_4 = self.v1_max_monthly_avg_summer_turbidity > 190
            si_1[mask_4] = 0

        if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_1)

    def calculate_si_2(self) -> np.ndarray:
        """Percent cover (vegetation, brush, debris, standing timber, etc.)
        during midsummer in pools, overflow areas, and backwaters"""
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()

        # set to ideal
        if self.v2_pct_cover_in_midsummer_pools_overflow_bw is None:
            self._logger.info(
                "Pct cover during midsummer in pools, overflow areas, and backwaters "
                "assumes ideal conditions. Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            # condition 1
            mask_1 = (
                self.v2_pct_cover_in_midsummer_pools_overflow_bw >= 25
            ) & (self.v2_pct_cover_in_midsummer_pools_overflow_bw < 85)
            si_2[mask_1] = 1

            # condition 2
            mask_2 = (
                self.v2_pct_cover_in_midsummer_pools_overflow_bw >= 0
            ) & (self.v2_pct_cover_in_midsummer_pools_overflow_bw < 25)
            si_2[mask_2] = (
                0.032
                * (self.v2_pct_cover_in_midsummer_pools_overflow_bw[mask_2])
            ) + 0.2

            # condition 3
            mask_3 = self.v2_pct_cover_in_midsummer_pools_overflow_bw >= 85
            si_2[mask_3] = (
                -0.04
                * (self.v2_pct_cover_in_midsummer_pools_overflow_bw[mask_3])
            ) + 4.37

        if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_2)

    def calculate_si_3(self) -> np.ndarray:
        """Stream gradient within study area"""
        self._logger.info("Running SI 3")
        si_3 = self.template.copy()

        # set to ideal
        if self.v3_stream_gradient is None:
            self._logger.info(
                "Stream gradient within study area assumes ideal conditions. Setting index to 1."
            )
            si_3[~np.isnan(si_3)] = 1

        else:
            # condition 1
            mask_1 = self.v3_stream_gradient < 0.5
            si_3[mask_1] = 1

            # condition 2
            mask_2 = (self.v3_stream_gradient < 2) & (
                self.v3_stream_gradient >= 0.5
            )
            si_3[mask_2] = (
                -0.6667 * (self.v3_stream_gradient[mask_2])
            ) + 1.3333

            # condition 3
            mask_3 = self.v3_stream_gradient >= 2
            si_3[mask_3] = 0

        if np.any(np.isclose(si_3, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_3)

    def calculate_si_4(self) -> np.ndarray:
        """Average current velocity in pools and backwater areas during average summer flow (Jul - Sept)"""
        self._logger.info("Running SI 4")
        si_4 = self.template.copy()

        if self.v4_avg_vel_summer_flow_pools_bw is None:
            self._logger.info(
                "Average current velocity in pools and backwater areas during average "
                "summer flow not provided. Setting index to 1."
            )
            si_4[~np.isnan(si_4)] = 1

        else:

            # condition 1
            mask_1 = self.v4_avg_vel_summer_flow_pools_bw < 10
            si_4[mask_1] = 1

            # condition 2
            mask_2 = (self.v4_avg_vel_summer_flow_pools_bw >= 10) & (
                self.v4_avg_vel_summer_flow_pools_bw < 15
            )
            si_4[mask_2] = (
                -0.06 * (self.v4_avg_vel_summer_flow_pools_bw[mask_2])
            ) + 1.6

            # condition 3
            mask_3 = (self.v4_avg_vel_summer_flow_pools_bw >= 15) & (
                self.v4_avg_vel_summer_flow_pools_bw < 60
            )
            si_4[mask_3] = (
                -0.0156 * (self.v4_avg_vel_summer_flow_pools_bw[mask_3])
            ) + 0.9333

            # condition 4
            mask_4 = self.v4_avg_vel_summer_flow_pools_bw >= 60
            si_4[mask_4] = 0

        if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_4)

    def calculate_si_5(self) -> np.ndarray:
        """Percent pools and backwater areas during average spring and summer flow (Apr - Sept)"""
        self._logger.info("Running SI 5")
        si_5 = self.template.copy()

        if self.v5_pct_pools_bw_avg_spring_summer_flow is None:
            self._logger.info(
                "Percent pools and backwater areas during average spring and summer flow not provided. Setting index to 1."
            )
            si_5[~np.isnan(si_5)] = 1

        else:

            # condition 1
            mask_1 = self.v5_pct_pools_bw_avg_spring_summer_flow <= 50
            si_5[mask_1] = 0.02 * (
                self.v5_pct_pools_bw_avg_spring_summer_flow[mask_1]
            )

            # condition 2
            mask_2 = self.v5_pct_pools_bw_avg_spring_summer_flow > 50
            si_5[mask_2] = 1

        if np.any(np.isclose(si_5, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_5)

    def calculate_si_6(self) -> np.ndarray:
        """No logic exists for si_6."""
        return NotImplementedError

    def calculate_si_7(self) -> np.ndarray:
        """pH levels during the year"""
        self._logger.info("Running SI 7")
        si_7 = self.template.copy()

        # set to ideal
        if self.v7_ph_year is None:
            self._logger.info(
                "pH leves during the year assumes ideal conditions. Setting index to 1."
            )
            si_7[~np.isnan(si_7)] = 1

        else:
            # condition 1
            mask_1 = (self.v7_ph_year <= 4.5) | (self.v7_ph_year > 9.5)
            si_7[mask_1] = 0

            # condition 2
            mask_2 = (self.v7_ph_year > 6.5) & (self.v7_ph_year <= 8.5)
            si_7[mask_2] = 1

            # condition 3
            mask_3 = (self.v7_ph_year > 4.5) & (self.v7_ph_year <= 6.5)
            si_7[mask_3] = 0.5 * (self.v7_ph_year[mask_3]) - 2.25

            # condition 4
            mask_4 = (self.v7_ph_year > 8.5) & (self.v7_ph_year <= 9.5)
            si_7[mask_4] = -1 * (self.v7_ph_year[mask_4]) + 9.5

            # condition 5
            mask_5 = self.v7_ph_year > 9.5
            si_7[mask_5] = 0

        if np.any(np.isclose(si_7, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_7)

    def calculate_si_8(self) -> np.ndarray:
        """Most suitable water temperature in pools and backwaters
        during midsummer (Jul - Aug) (adult)"""
        self._logger.info("Running SI 8")
        si_8 = self.template.copy()

        if self.v8_most_suit_temp_in_midsummer_pools_bw_adult is None:
            self._logger.info(
                "Most suitable water temperature in pools and backwaters during midsummer (adult)"
                "is not provided. Setting index to 1."
            )
            si_8[~np.isnan(si_8)] = 1

        else:
            # condition 1
            mask_1 = (
                self.v8_most_suit_temp_in_midsummer_pools_bw_adult <= 14
            ) | (self.v8_most_suit_temp_in_midsummer_pools_bw_adult > 34)
            si_8[mask_1] = 0

            # condition 2
            mask_2 = (
                self.v8_most_suit_temp_in_midsummer_pools_bw_adult > 23
            ) & (self.v8_most_suit_temp_in_midsummer_pools_bw_adult <= 27)
            si_8[mask_2] = 1

            # condition 3
            mask_3 = (
                self.v8_most_suit_temp_in_midsummer_pools_bw_adult > 14
            ) & (self.v8_most_suit_temp_in_midsummer_pools_bw_adult <= 23)
            si_8[mask_3] = (
                0.1112
                * (self.v8_most_suit_temp_in_midsummer_pools_bw_adult[mask_3])
                - 1.5659
            )

            # condition 4
            mask_4 = (
                self.v8_most_suit_temp_in_midsummer_pools_bw_adult > 27
            ) & (self.v8_most_suit_temp_in_midsummer_pools_bw_adult <= 34)
            si_8[mask_4] = (
                -0.143
                * (self.v8_most_suit_temp_in_midsummer_pools_bw_adult[mask_4])
                + 4.86
            )

        # TODO: apply backwater mask here? (either set to NaN or set to 0?)

        if np.any(np.isclose(si_8, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_8)

    def calculate_si_9(self) -> np.ndarray:
        """Most suitable water temperature in pools and backwaters
        during midsummer (Jul - Aug) (juvenile)"""
        self._logger.info("Running SI 9")
        si_9 = self.template.copy()

        if self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile is None:
            self._logger.info(
                "Most suitable water temperature in pools and backwaters during midsummer (juvenile)"
                "is not provided. Setting index to 1."
            )
            si_9[~np.isnan(si_9)] = 1

        else:
            # condition 1
            mask_1 = (
                self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile <= 11
            ) | (self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile >= 30)
            si_9[mask_1] = 0

            # condition 2
            mask_2 = (
                self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile > 22
            ) & (self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile <= 24)
            si_9[mask_2] = 1

            # condition 3
            mask_3 = (
                self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile > 11
            ) & (self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile <= 22)
            si_9[mask_3] = (
                0.0909
                * (
                    self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile[
                        mask_3
                    ]
                )
                - 1
            )

            # condition 4
            mask_4 = (
                self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile > 24
            ) & (self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile < 30)
            si_9[mask_4] = (
                -0.1667
                * (
                    self.v9_most_suit_temp_in_midsummer_pools_bw_juvenile[
                        mask_4
                    ]
                )
                + 5
            )

        if np.any(np.isclose(si_9, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_9)

    def calculate_si_10(self) -> np.ndarray:
        """Average water temperature in pools and backwaters during midsummer (fry)"""
        self._logger.info("Running SI 10")
        si_10 = self.template.copy()

        if self.v10_avg_midsummer_temp_in_pools_bw_fry is None:
            self._logger.info(
                "Avg water temp in pools and backwaters during midsummer (fry)"
                "is not provided. Setting index to 1."
            )
            si_10[~np.isnan(si_10)] = 1

        else:
            # condition 1
            mask_1 = (self.v10_avg_midsummer_temp_in_pools_bw_fry <= 12) | (
                self.v10_avg_midsummer_temp_in_pools_bw_fry > 30
            )
            si_10[mask_1] = 0

            # condition 2
            mask_2 = (self.v10_avg_midsummer_temp_in_pools_bw_fry > 20) & (
                self.v10_avg_midsummer_temp_in_pools_bw_fry <= 24
            )
            si_10[mask_2] = 1

            # condition 3
            mask_3 = (self.v10_avg_midsummer_temp_in_pools_bw_fry > 12) & (
                self.v10_avg_midsummer_temp_in_pools_bw_fry <= 15
            )
            si_10[mask_3] = (
                0.066 * (self.v10_avg_midsummer_temp_in_pools_bw_fry[mask_3])
            ) - 0.8

            # condition 4
            mask_4 = (self.v10_avg_midsummer_temp_in_pools_bw_fry > 15) & (
                self.v10_avg_midsummer_temp_in_pools_bw_fry <= 20
            )
            si_10[mask_4] = (
                0.16 * (self.v10_avg_midsummer_temp_in_pools_bw_fry[mask_4])
            ) - 2.2021

            # condition 5
            mask_5 = (self.v10_avg_midsummer_temp_in_pools_bw_fry > 24) & (
                self.v10_avg_midsummer_temp_in_pools_bw_fry <= 30
            )
            si_10[mask_5] = (
                -0.167 * (self.v10_avg_midsummer_temp_in_pools_bw_fry[mask_5])
            ) + 5

        if np.any(np.isclose(si_10, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_10)

    def calculate_si_11(self) -> np.ndarray:
        """Average water temperature in backwaters during spawning (embryo) (Feb - Mar)"""
        self._logger.info("Running SI 11")
        si_11 = self.template.copy()

        if self.v11_avg_spawning_temp_in_bw_embryo is None:
            self._logger.info(
                "Average water temperature in backwaters during spawning (embryo) is not provided"
                "Setting index to 1."
            )
            si_11[~np.isnan(si_11)] = 1

        else:
            # condition 1
            mask_1 = (self.v11_avg_spawning_temp_in_bw_embryo <= 12) | (
                self.v11_avg_spawning_temp_in_bw_embryo >= 23
            )
            si_11[mask_1] = 0

            # condition 2
            mask_2 = (self.v11_avg_spawning_temp_in_bw_embryo >= 17) & (
                self.v11_avg_spawning_temp_in_bw_embryo <= 20
            )
            si_11[mask_2] = 1

            # condition 3
            mask_3 = (self.v11_avg_spawning_temp_in_bw_embryo > 12) & (
                self.v11_avg_spawning_temp_in_bw_embryo < 17
            )
            si_11[mask_3] = (
                0.2 * (self.v11_avg_spawning_temp_in_bw_embryo[mask_3])
            ) - 2.4

            # condition 4
            mask_4 = (self.v11_avg_spawning_temp_in_bw_embryo > 20) & (
                self.v11_avg_spawning_temp_in_bw_embryo < 23
            )
            si_11[mask_4] = (
                -0.33 * (self.v11_avg_spawning_temp_in_bw_embryo[mask_4])
            ) + 7.7

        if np.any(np.isclose(si_11, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_11)

    def calculate_si_12(self) -> np.ndarray:
        """Minimum dissolved oxygen levels within temperature strata selected above
        (V8, V9, and V10) during midsummer (adult, juvenile, fry) (Jul - Aug)
        """
        self._logger.info("Running SI 12")
        si_12 = self.template.copy()

        if self.v12_min_do_in_midsummer_temp_strata is None:
            self._logger.info(
                "Minimum dissolved oxygen levels within temperature strata during midsummer"
                "(adult, juvenile, fry) is not provided. Setting index to 1."
            )
            si_12[~np.isnan(si_12)] = 1

        else:
            # condition 1
            mask_1 = self.v12_min_do_in_midsummer_temp_strata <= 1.5
            si_12[mask_1] = 0

            # condition 2
            mask_2 = self.v12_min_do_in_midsummer_temp_strata >= 5

            si_12[mask_2] = 1

            # condition 3
            mask_3 = (self.v12_min_do_in_midsummer_temp_strata > 1.5) & (
                self.v12_min_do_in_midsummer_temp_strata <= 4.5
            )
            si_12[mask_3] = (
                0.066 * self.v12_min_do_in_midsummer_temp_strata[mask_3] - 0.1
            )

            # condition 4
            mask_4 = (self.v12_min_do_in_midsummer_temp_strata > 4.5) & (
                self.v12_min_do_in_midsummer_temp_strata < 5
            )
            si_12[mask_4] = (
                1.59 * self.v12_min_do_in_midsummer_temp_strata[mask_4] - 6.9
            )

        if np.any(np.isclose(si_12, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_12)

    def calculate_si_13(self) -> np.ndarray:
        """Minimum dissolved oxygen levels within backwaters during spawning (embryo, fry)"""
        self._logger.info("Running SI 13")
        si_13 = self.template.copy()

        if self.v13_min_do_in_spawning_bw is None:
            self._logger.info(
                "Minimum dissolved oxygen levels within backwaters during spawning (embryo, fry) "
                "is not provided. Setting index to 1."
            )
            si_13[~np.isnan(si_13)] = 1

        else:
            # condition 1
            mask_1 = self.v13_min_do_in_spawning_bw < 2.7
            si_13[mask_1] = 0

            # condition 2
            mask_2 = self.v13_min_do_in_spawning_bw >= 5
            si_13[mask_2] = 1

            # condition 3
            mask_3 = (self.v13_min_do_in_spawning_bw >= 2.7) & (
                self.v13_min_do_in_spawning_bw < 3.5
            )
            si_13[mask_3] = 0.5 * self.v13_min_do_in_spawning_bw[mask_3] - 1.35

            # condition 4
            mask_4 = (self.v13_min_do_in_spawning_bw >= 3.5) & (
                self.v13_min_do_in_spawning_bw < 5
            )
            si_13[mask_4] = 0.4 * self.v13_min_do_in_spawning_bw[mask_4] - 1

        if np.any(np.isclose(si_13, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_13)

    def calculate_si_14(self) -> np.ndarray:
        """Maximum salinity during growing season (Apr - Sept)"""
        self._logger.info("Running SI 14")
        si_14 = self._create_template_array(self.v14_max_salinity_gs)

        if self.v14_max_salinity_gs is None:
            self._logger.info(
                "Maximum salinity during growing season is not provided. Setting index to 1."
            )
            si_14[~np.isnan(si_14)] = 1

        else:
            # condition 1
            mask_1 = self.v14_max_salinity_gs < 0.1
            si_14[mask_1] = 10 * self.v14_max_salinity_gs[mask_1]

            # condition 2
            mask_2 = (self.v14_max_salinity_gs >= 0.1) & (
                self.v14_max_salinity_gs < 2
            )
            si_14[mask_2] = 1

            # condition 3
            mask_3 = (self.v14_max_salinity_gs >= 2) & (
                self.v14_max_salinity_gs < 4.7
            )
            si_14[mask_3] = -0.2963 * self.v14_max_salinity_gs[mask_3] + 1.5926

            # condition 4
            mask_4 = (self.v14_max_salinity_gs >= 4.7) & (
                self.v14_max_salinity_gs < 5
            )
            si_14[mask_4] = -0.6667 * self.v14_max_salinity_gs[mask_4] + 3.3333

            # condition 5
            mask_5 = self.v14_max_salinity_gs >= 5
            si_14[mask_5] = 0

        if np.any(np.isclose(si_14, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_14)

    def calculate_si_15(self) -> np.ndarray:
        """No logic exists for si_15."""
        return NotImplementedError

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Black Crappie Riverine final HSI.")
        hsi = self.template.copy()
        for si_name, si_array in [
            ("SI 1", self.si_1),
            ("SI 2", self.si_2),
            ("SI 4", self.si_4),
            ("SI 5", self.si_5),
            ("SI 7", self.si_7),
            ("SI 8", self.si_8),
            ("SI 9", self.si_9),
            ("SI 10", self.si_10),
            ("SI 11", self.si_11),
            ("SI 12", self.si_12),
            ("SI 13", self.si_13),
            ("SI 14", self.si_14),
        ]:
            invalid_values = (si_array < 0) | (si_array > 1)
            if np.any(invalid_values):
                num_invalid = np.count_nonzero(invalid_values)
                self._logger.warning(
                    "%s contains %d values outside the range [0, 1].",
                    si_name,
                    num_invalid,
                )

        # individual riverine model components
        # food cover component (fc)
        self.fc = (self.si_2 * self.si_5) ** (1 / 2)

        # water quality component (wq)
        # water quality term for cube root
        self.wq_tcr = (self.si_8 * self.si_9 * self.si_10) ** (1 / 3)

        # condition 1 (water quality condition for cube root term)
        wq_tcr_mask = (
            (self.si_8 <= 0.4) | (self.si_9 <= 0.4) | (self.si_10 <= 0.4)
        )
        self.wq_tcr_adj = np.where(
            wq_tcr_mask,
            np.minimum.reduce([self.si_8, self.si_9, self.si_10]),
            self.wq_tcr,
        )

        # water quality initial equation
        self.wq_init = (
            2 * (self.wq_tcr_adj)
            + 2 * (self.si_12)
            + self.si_7
            + self.si_1
            + self.si_14
        ) / 7

        # condition 2
        wq_mask = ((self.wq_tcr_adj) <= 0.4) | (self.si_12 <= 0.4)
        self.wq = np.where(
            wq_mask,
            np.minimum.reduce([self.wq_tcr_adj, self.si_12, self.wq_init]),
            self.wq_init,
        )

        # reproduction component (rc)
        self.rc = (
            (self.si_2)
            * (self.si_5)
            * (self.si_11 ** (2))
            * (self.si_13 ** (2))
        ) ** (1 / 6)

        # other component (ot)
        self.ot = (self.si_3 + self.si_4) / 2

        # Combine individual suitability indices
        initial_hsi = (self.fc * self.wq * self.rc * self.ot) ** (1 / 4)

        # If wq or rc <= 0.4, select min(wq, rc, initial_hsi)
        mask_hsi = (self.wq <= 0.4) | (self.rc <= 0.4)
        hsi = np.where(
            mask_hsi,
            np.minimum.reduce([self.wq, self.rc, initial_hsi]),
            initial_hsi,
        )

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
