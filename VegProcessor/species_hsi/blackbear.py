from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class BlackBearHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    hydro_domain_480: np.ndarray = None
    dem_480: np.ndarray = None

    v1_pct_area_wetland_cover: np.ndarray = None
    v2_pct_canopy_cover_s_mast_species: np.ndarray = None
    v3_soft_mast_prod_species_above1pct_canopy: np.ndarray = None
    v4_basal_area_mast_prod_species_above_30yr: np.ndarray = None
    v5_num_h_mast_species_w_one_mature_tree: np.ndarray = None
    v6_pct_area_nonforested_cover_250m: np.ndarray = None
    v7_pct_cover_over1pct_cover_h_mast_species: np.ndarray = None
    v8_pct_eval_area_inside_zones: np.ndarray = None

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)
    si_3: np.ndarray = field(init=False)
    si_4: np.ndarray = field(init=False)
    si_5: np.ndarray = field(init=False)
    si_6: np.ndarray = field(init=False)
    si_7: np.ndarray = field(init=False)
    si_8: np.ndarray = field(init=False)

    # components and equations
    spring_food: np.ndarray = field(init=False)
    summer_food: np.ndarray = field(init=False)
    fall_food: np.ndarray = field(init=False)
    human_intolerance: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create BlackBearHSI instance from an HSI instance."""

        def safe_divide(array: np.ndarray, divisor: int = 100) -> np.ndarray:
            """Helper function to divide arrays when decimal values are required
            by the SI logic. In the case of None (no array) it is preserved and
            passed to SI methods."""
            return array / divisor if array is not None else None

        return cls(
            v1_pct_area_wetland_cover=hsi_instance.pct_vegetated,
            v2_pct_canopy_cover_s_mast_species=hsi_instance.pct_soft_mast,
            v3_soft_mast_prod_species_above1pct_canopy=hsi_instance.num_soft_mast_species,  # set to ideal
            v4_basal_area_mast_prod_species_above_30yr=hsi_instance.basal_area_hard_mast,  # set to ideal
            v5_num_h_mast_species_w_one_mature_tree=hsi_instance.num_hard_mast_species,  # set to ideal
            v6_pct_area_nonforested_cover_250m=hsi_instance.pct_near_forest,
            v7_pct_cover_over1pct_cover_h_mast_species=hsi_instance.pct_cover_hard_mast,
            v8_pct_eval_area_inside_zones=hsi_instance.pct_area_zone_influence,
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
        self.si_5 = self.calculate_si_5()
        self.si_6 = self.calculate_si_6()
        self.si_7 = self.calculate_si_7()
        self.si_8 = self.calculate_si_8()

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _create_template_array(self) -> np.ndarray:
        """Create an array from a template all valid pixels are 999.0, and
        NaN from the input are persisted.
        """
        # black bear does not have depth related vars, and is therefore not
        # limited to hydrologic model domain
        arr = np.where(np.isnan(self.dem_480), np.nan, 999.0)
        # arr = np.full(self.v6_pct_cell_open_water.shape, 999.0)
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
        self._logger = logging.getLogger("BlackBearHSI")
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
        """Percent of area in wetland cover types (excluding open water)"""
        self._logger.info("Running SI 1")
        si_1 = self.template.copy()

        # self.v1_pct_area_wetland_cover is None:
        if self.v1_pct_area_wetland_cover is None:
            self._logger.info(
                "Pct of area in wetland cover types (excluding open water) not provided. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else:
            # Note: equations use % values not decimals
            # self.v1_pct_area_wetland_cover /= 100
            # condition 1
            mask_1 = self.v1_pct_area_wetland_cover < 7.0
            si_1[mask_1] = (
                0.0714 * self.v1_pct_area_wetland_cover[mask_1]
            ) + 0.5

            # condition 2
            mask_2 = (self.v1_pct_area_wetland_cover >= 7.0) & (
                self.v1_pct_area_wetland_cover <= 50
            )
            si_1[mask_2] = 1.0

            # condition 3
            mask_3 = self.v1_pct_area_wetland_cover > 50
            si_1[mask_3] = (
                -0.01 * self.v1_pct_area_wetland_cover[mask_3]
            ) + 1.5

        if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_1)

    def calculate_si_2(self) -> np.ndarray:
        """Percent canopy cover of soft mast producing species (includes hazel)."""
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()

        if self.v2_pct_canopy_cover_s_mast_species is None:
            self._logger.info(
                "Pct canopy cover of soft mast producing species not provided. Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            # condition 1
            mask_1 = self.v2_pct_canopy_cover_s_mast_species < 25.0
            si_2[mask_1] = (
                0.036 * self.v2_pct_canopy_cover_s_mast_species[mask_1]
            ) + 0.1

            # condition 2
            mask_2 = self.v2_pct_canopy_cover_s_mast_species >= 25.0
            si_2[mask_2] = 1.0

        if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_2)

    def calculate_si_3(self) -> np.ndarray:
        """Number of soft mast producing species present at ≥ 1% canopy cover."""
        self._logger.info("Running SI 3")
        si_3 = self.template.copy()

        # Set to ideal. Not tracking number of species
        if self.v3_soft_mast_prod_species_above1pct_canopy is None:
            self._logger.info(
                "Number of soft mast producing species present at ≥ 1pct canopy cover assumes "
                "ideal conditions. Setting index to 1."
            )
            si_3[~np.isnan(si_3)] = 1

        else:
            raise NotImplementedError(
                "No logic for black bear v3 exists. Either use ideal (set array None) or add logic."
            )

        if np.any(np.isclose(si_3, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_3)

    def calculate_si_4(self) -> np.ndarray:
        """Basal area of hard mast producing species ≥ 30 years in age."""
        self._logger.info("Running SI 4")
        si_4 = self.template.copy()

        # Set to ideal
        if self.v4_basal_area_mast_prod_species_above_30yr is None:
            self._logger.info(
                "Basal area of hard mast producing species ≥ 30 years assumes ideal conditions. "
                "Setting index to 1."
            )
            si_4[~np.isnan(si_4)] = 1

        else:
            raise NotImplementedError(
                "No logic for black bear v4 exists. Either use ideal (set array None) or add logic."
            )

            # if ft2/ac
            # else:
            #    # condition 1 (basal area units in ft2/ac)
            #    mask_1 = self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr < 70
            #    si_4[mask_1] = (0.0129 * self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr[mask_1]) + 0.1

            #   # condition 2 (basal area units in ft2/ac)
            #    mask_2 = (self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr >= 70) & (
            #        self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr <= 90)
            #    si_4[mask_2] = 1

            #    #condition 3 (basal area units in ft2/ac)
            #    mask_3 = (self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr) > 90
            #    si_4[mask_3] = (-0.01 * self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr[mask_3]) + 1.9

            #    if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
            #        raise ValueError("Unhandled condition in SI logic!")

            # if self.hydro_domain_flag:
            #     si_4 = np.where(~np.isnan(self.hydro_domain_480), si_4, np.nan)

            # if m2/0.4ha
            # else:
            #    # condition 1 (basal area units in m2/0.4ha)
            #    mask_1 = self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr < 6.5
            #    si_4[mask_1] = (0.1385 * self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr[mask_1]) + 0.1

            #    # condition 2 (basal area units in m2/0.4ha)
            #    mask_2 = (self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr >= 6.5) & (
            #        self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr <= 8.4)
            #    si_4[mask_2] = 1

            #    #condition 3 (basal area units in m2/0.4ha)
            #    mask_3 = (self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr) > 8.4
            #    si_4[mask_3] = (-0.1111 * self.v4_basal_area_mast_prod_species_greaterthanorequalto_30yr[mask_3]) + 1.9333

        if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_4)

    def calculate_si_5(self) -> np.ndarray:
        """Number of hard mast producing species present with at least one mature tree per 0.4 ha."""
        self._logger.info("Running SI 5")
        si_5 = self.template.copy()

        # Set to ideal. Not tracking number of species
        if self.v5_num_h_mast_species_w_one_mature_tree is None:
            self._logger.info(
                "Num of hard mast producing species present with at least one mature tree per 0.4 ha asssumes ideal conditions. "
                "Setting index to 1."
            )
            si_5[~np.isnan(si_5)] = 1

        else:
            raise NotImplementedError(
                "No logic for black bear v5 exists. Either use ideal (set array None) or add logic."
            )

        if np.any(np.isclose(si_5, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_5)

    def calculate_si_6(self) -> np.ndarray:
        """Percent of area in nonforested cover types ≤ 250m from forested cover types."""
        self._logger.info("Running SI 6")
        si_6 = self.template.copy()

        if self.v6_pct_area_nonforested_cover_250m is None:
            self._logger.info(
                "Pct of area in nonforested cover types ≤ 250m from forested cover types not provided. Setting index to 1."
            )
            si_6[~np.isnan(si_6)] = 1

        else:
            # condition 1
            mask_1 = self.v6_pct_area_nonforested_cover_250m < 25
            si_6[mask_1] = (
                0.032 * self.v6_pct_area_nonforested_cover_250m[mask_1]
            ) + 0.2

            # condition 2
            mask_2 = (self.v6_pct_area_nonforested_cover_250m >= 25) & (
                self.v6_pct_area_nonforested_cover_250m <= 50
            )
            si_6[mask_2] = (
                -0.04 * self.v6_pct_area_nonforested_cover_250m[mask_2]
            ) + 3

            # condition 3
            mask_3 = self.v6_pct_area_nonforested_cover_250m > 75
            si_6[mask_3] = 0

        if np.any(np.isclose(si_6, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_6)

    def calculate_si_7(self) -> np.ndarray:
        """Percent of area in cover types that have ≥ 1pct cover of hard mast producing species."""
        self._logger.info("Running SI 7")
        si_7 = self.template.copy()

        if self.v7_pct_cover_over1pct_cover_h_mast_species is None:
            self._logger.info(
                "Pct of area in cover types that have ≥ 1pct cover of hard mast producing species not provided. Setting index to 1."
            )
            si_7[~np.isnan(si_7)] = 1

        else:
            # condition 1
            mask_1 = self.v7_pct_cover_over1pct_cover_h_mast_species < 35
            si_7[mask_1] = (
                0.0257
                * self.v7_pct_cover_over1pct_cover_h_mast_species[mask_1]
            ) + 0.1

            # condition 2
            mask_2 = (
                self.v7_pct_cover_over1pct_cover_h_mast_species >= 35
            ) & (self.v7_pct_cover_over1pct_cover_h_mast_species <= 100)
            si_7[mask_2] = 1.0

        if np.any(np.isclose(si_7, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_7)

    def calculate_si_8(self) -> np.ndarray:
        """Percent of evaluation area inside of zones of influence defined by radii 5.7 km around towns;
        3.5 km around cropland; and 1.1 km around residences."""
        ## Calculate for inital conditions and use for all time periods
        self._logger.info("Running SI 8")
        si_8 = self.template.copy()

        if self.v8_pct_eval_area_inside_zones is None:
            self._logger.info(
                "Pct of eval area inside of zones of influence not provided. Setting index to 1."
            )
            si_8[~np.isnan(si_8)] = 1

        else:
            # condition 1 (only one condition so no mask needed)
            si_8 = (-0.0095 * self.v8_pct_eval_area_inside_zones) + 1

        if np.any(np.isclose(si_8, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_8)

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Black Bear final HSI.")
        for si_name, si_array in [
            ("SI 1", self.si_1),
            ("SI 2", self.si_2),
            ("SI 3", self.si_3),
            ("SI 4", self.si_4),
            ("SI 5", self.si_5),
            ("SI 6", self.si_6),
            ("SI 7", self.si_7),
            ("SI 8", self.si_8),
        ]:
            invalid_values = (si_array < 0) | (si_array > 1)
            if np.any(invalid_values):
                num_invalid = np.count_nonzero(invalid_values)
                self._logger.warning(
                    "%s contains %d values outside the range [0, 1].",
                    si_name,
                    num_invalid,
                )

        spring_food = self.si_1
        summer_food = (self.si_2 * self.si_3) ** (1 / 2)
        fall_food = (self.si_3 * self.si_4) ** (1 / 2)
        human_intolerance = self.si_8

        # Combine individual suitability indices
        hsi = (
            (spring_food + (summer_food * self.si_6) + (fall_food * self.si_7))
            / 3
        ) * human_intolerance

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
