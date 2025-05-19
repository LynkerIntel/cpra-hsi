from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class SwampHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    hydro_domain_480: np.ndarray = None
    dem_480: np.ndarray = None
    pct_swamp_bottom_hardwood: np.ndarray = None
    pct_zone_ii: np.ndarray = None

    v1a_pct_overstory: np.ndarray = None
    v1b_pct_midstory: np.ndarray = None
    v1c_pct_understory: np.ndarray = None
    v2_maturity_dbh: np.ndarray = None
    v3a_flood_duration: np.ndarray = None
    v3b_flow_exchange: np.ndarray = None
    v4_mean_high_salinity_gs: np.ndarray = None
    v5_forested_connectivity_cat: np.ndarray = None
    v6_suit_trav_surr_lu: np.ndarray = None
    v7_disturbance: np.ndarray = None

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
        """Create Swamp HSI instance from an HSI instance."""

        return cls(
            v1a_pct_overstory=hsi_instance.pct_overstory,
            v1b_pct_midstory=hsi_instance.pct_midstory,
            v1c_pct_understory=hsi_instance.pct_understory,
            v2_maturity_dbh=hsi_instance.maturity_dbh,  # set to ideal
            v3a_flood_duration=hsi_instance.flood_duration,
            v3b_flow_exchange=hsi_instance.flow_exchange,
            v4_mean_high_salinity_gs=hsi_instance.mean_high_salinity_gs,
            v5_forested_connectivity_cat=hsi_instance.forested_connectivity_cat,
            v6_suit_trav_surr_lu=hsi_instance.suit_trav_surr_lu,  # set to ideal
            v7_disturbance=hsi_instance.disturbance,  # set to ideal
            dem_480=hsi_instance.dem_480,
            hydro_domain_480=hsi_instance.hydro_domain_480,
            pct_swamp_bottom_hardwood=hsi_instance.pct_swamp_bottom_hardwood,
            pct_zone_ii=hsi_instance.pct_zone_ii,
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

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _create_template_array(self) -> np.ndarray:
        """Create an array from a template all valid pixels are 999.0, and
        NaN from the input are persisted.
        """
        # Swamp Wetland Value Assessment (Swamp WVA) has depth related vars, and is
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
        self._logger = logging.getLogger("SwampHSI")
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

    def swamp_blh_mask(self, si_array: np.ndarray) -> np.ndarray:
        """To apply the Swamp WVA, at least 33% forest cover (Zone II to V)
        has to be present of which greater than 60% is in Zone II.
        This applies a mask to each SI array where these conditions are not met.
        These areas are given an SI = 0.
        """
        if self.pct_swamp_bottom_hardwood is not None:
            swamp_blh_mask = (
                (self.pct_swamp_bottom_hardwood < 33)
                & (self.pct_zone_ii < 60)
                & (~np.isnan(self.pct_swamp_bottom_hardwood))
            )
            si_array[swamp_blh_mask] = 0
        return si_array

    def calculate_si_1(self) -> np.ndarray:
        """Stand Structure"""
        self._logger.info("Running SI 1")
        si_1 = self.template.copy()

        if (
            self.v1a_pct_overstory is None
            or self.v1b_pct_midstory is None
            or self.v1c_pct_understory is None
        ):
            self._logger.info(
                "Stand structure data not provided. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else:

            # assign intermediate masks with "or equal to"
            # default condition is greater than, which
            # can be inverted with "~"
            mask_overs_33 = self.v1a_pct_overstory >= 33
            mask_overs_50 = self.v1a_pct_overstory >= 50
            mask_overs_75 = self.v1a_pct_overstory >= 75
            mask_mids_33 = self.v1b_pct_midstory >= 33
            mask_unders_33 = self.v1c_pct_understory >= 33

            # class 1: overstory < 33
            class_1 = self.v1a_pct_overstory < 33
            si_1[class_1] = 0.1

            # class 2: 33–50 overstory & mid < 33 & under < 33
            class_2 = (
                (mask_overs_33 & (self.v1a_pct_overstory < 50))
                & ~mask_mids_33
                & ~mask_unders_33
            )
            si_1[class_2] = 0.2

            # class 3:
            # block 1: 33-50 overstory & (mid >= 33 or under >= 33)
            # block 2: 50-75 overstory & mid < 33 & under < 33
            class_3_block1 = (
                mask_overs_33 & (self.v1a_pct_overstory < 50)
            ) & (mask_mids_33 | mask_unders_33)

            class_3_block2 = (
                (mask_overs_50 & (self.v1a_pct_overstory < 75))
                & ~mask_mids_33
                & ~mask_unders_33
            )
            class_3 = class_3_block1 | class_3_block2
            si_1[class_3] = 0.4

            # class 4:
            # block 1: 50-75 overstory & (mid >= 33 or under >= 33)
            # block 2: >= 75 overstory & mid < 33 or under < 33
            class_4_block1 = (
                mask_overs_50 & (self.v1a_pct_overstory < 75)
            ) & (mask_mids_33 | mask_unders_33)

            class_4_block2 = mask_overs_75 & ~mask_mids_33 & ~mask_unders_33

            class_4 = class_4_block1 | class_4_block2
            si_1[class_4] = 0.6

            # class 5: 33–50 & mid ≥ 33 & under ≥ 33
            class_5 = (
                (mask_overs_33 & (self.v1a_pct_overstory < 50))
                & mask_mids_33
                & mask_unders_33
            )
            si_1[class_5] = 0.8

            # class 6
            # block 1: >= 50 overstory & mid >= 33 & under >= 33
            # block 2: >= 75 overstory & (mid >= 33 or under >= 33)
            class_6_block1 = mask_overs_50 & mask_mids_33 & mask_unders_33
            class_6_block2 = mask_overs_75 & (mask_mids_33 | mask_unders_33)

            class_6 = class_6_block1 | class_6_block2
            si_1[class_6] = 1.0

        si_1 = self.swamp_blh_mask(si_1)

        if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_1)

    def calculate_si_2(self) -> np.ndarray:
        """Stand maturity based on diameter at breast height (dbh)"""
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()

        if self.v2_maturity_dbh is None:
            self._logger.info(
                "Stand maturity assumes ideal conditions. Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            raise NotImplementedError(
                "No logic for swamp v2 exists. Either use ideal (set array None) or add logic."
            )

        si_2 = self.swamp_blh_mask(si_2)

        if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_2)

    def calculate_si_3(self) -> np.ndarray:
        """Hydrology"""
        self._logger.info("Running SI 3")
        si_3 = self.template.copy()

        if self.v3a_flood_duration is None or self.v3b_flow_exchange is None:
            self._logger.info(
                "Flood duration or flow exchange data is not provided. Setting index to 1."
            )
            si_3[~np.isnan(si_3)] = 1

        else:
            # scoring for 20 flood duration and
            # flow exchange combinations of
            # conditions
            si3_score = {
                ("None", "High"): 0.9,
                ("Temporary", "High"): 0.9,
                ("Seasonal", "High"): 1.00,
                ("Semi-Permanent", "High"): 0.75,
                ("Permanent", "High"): 0.65,
                ("None", "Moderate"): 0.75,
                ("Temporary", "Moderate"): 0.75,
                ("Seasonal", "Moderate"): 0.85,
                ("Semi-Permanent", "Moderate"): 0.65,
                ("Permanent", "Moderate"): 0.45,
                ("None", "Low"): 0.65,
                ("Temporary", "Low"): 0.65,
                ("Seasonal", "Low"): 0.7,
                ("Semi-Permanent", "Low"): 0.45,
                ("Permanent", "Low"): 0.3,
                ("None", "None"): 0.4,
                ("Temporary", "None"): 0.4,
                ("Seasonal", "None"): 0.5,
                ("Semi-Permanent", "None"): 0.25,
                ("Permanent", "None"): 0.1,
            }

            # define conditions and scores
            conds = []
            scoring = []

            for (flood_dur, flow_exch), score in si3_score.items():
                mask = (self.v3a_flood_duration == flood_dur) & (
                    self.v3b_flow_exchange == flow_exch
                )
                conds.append(mask)
                scoring.append(score)

            si_3 = np.select(conds, scoring, default=si_3)

        si_3 = self.swamp_blh_mask(si_3)

        if np.any(np.isclose(si_3, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_3)

    def calculate_si_4(self) -> np.ndarray:
        """Mean High Salinity During the Growing Season (March to Oct)"""
        self._logger.info("Running SI 4")
        si_4 = self.template.copy()

        if self.v4_mean_high_salinity_gs is None:
            self._logger.info(
                "Mean high salinity is not provided. Setting index to 1."
            )
            si_4[~np.isnan(si_4)] = 1

        else:
            # condition 1
            mask_1 = (self.v4_mean_high_salinity_gs >= 0) & (
                self.v4_mean_high_salinity_gs <= 1
            )
            si_4[mask_1] = 1

            # condition 2
            mask_2 = (self.v4_mean_high_salinity_gs > 1) & (
                self.v4_mean_high_salinity_gs < 3
            )
            si_4[mask_2] = (
                -0.45 * self.v4_mean_high_salinity_gs[mask_2]
            ) + 1.45

            # condition 3
            mask_3 = self.v4_mean_high_salinity_gs >= 3
            si_4[mask_3] = 0.1

        si_4 = self.swamp_blh_mask(si_4)

        if np.any(np.isclose(si_4, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_4)

    def calculate_si_5(self) -> np.ndarray:
        """Size of Contiguous Forested Area in Acres"""
        self._logger.info("Running SI 5")
        si_5 = self.template.copy()

        if self.v5_forested_connectivity_cat is None:
            self._logger.info(
                "Size of contiguous forested area in acres not provided. Setting index to 1."
            )
            si_5[~np.isnan(si_5)] = 1

        else:
            mask_1 = self.v5_forested_connectivity_cat == 1
            si_5[mask_1] = 0.2

            mask_2 = self.v5_forested_connectivity_cat == 2
            si_5[mask_2] = 0.4

            mask_3 = self.v5_forested_connectivity_cat == 3
            si_5[mask_3] = 0.6

            mask_4 = self.v5_forested_connectivity_cat == 4
            si_5[mask_4] = 0.8

            mask_5 = self.v5_forested_connectivity_cat == 5
            si_5[mask_5] = 1

            # Areas with a DBH less than 5 are excluded from further logic
            if self.v2_maturity_dbh is not None:
                self._logger.info("DBH is < 5. Setting index to 1.")
                dbh_mask = self.v2_maturity_dbh < 5
                si_5[dbh_mask] = 1

            else:
                self._logger.warning(
                    "Stand maturity (dbh) not provided. All areas are included in logic."
                )

        si_5 = self.swamp_blh_mask(si_5)

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
                "No logic for swamp v6 exists. Either use ideal (set array None) or add logic."
            )

        si_6 = self.swamp_blh_mask(si_6)

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
                "No logic for swamp v7 exists. Either use ideal (set array None) or add logic."
            )

        si_7 = self.swamp_blh_mask(si_7)

        if np.any(np.isclose(si_7, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_7)

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Swamp WVA final HSI.")
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
        hsi = (
            (self.si_1**3.0)
            * (self.si_2**2.5)
            * (self.si_3**3.0)
            * (self.si_4**1.5)
            * (self.si_5)
            * (self.si_6)
            * (self.si_7)
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
