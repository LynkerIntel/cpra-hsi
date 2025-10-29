from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class BottomlandHardwoodHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    hydro_domain_480: np.ndarray = None
    dem_480: np.ndarray = None
    pct_blh_cover: np.ndarray = None

    v1a_pct_overstory_w_mast: np.ndarray = None
    v1b_pct_overstory_w_soft_mast: np.ndarray = None
    v1c_pct_overstory_w_hard_mast: np.ndarray = None
    v2_stand_maturity: np.ndarray = None
    v3a_pct_understory: np.ndarray = None
    v3b_pct_midstory: np.ndarray = None
    v4a_flood_duration: np.ndarray = None
    v4b_flow_exchange: np.ndarray = None
    v5_forested_connectivity_cat: np.ndarray = None

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
        """Create Bottomland Hardwood (BLH) HSI instance from an HSI instance."""

        return cls(
            v1a_pct_overstory_w_mast=hsi_instance.pct_has_mast,
            v1b_pct_overstory_w_soft_mast=hsi_instance.pct_soft_mast,
            v1c_pct_overstory_w_hard_mast=hsi_instance.pct_hard_mast,
            v2_stand_maturity=hsi_instance.maturity_480,
            v3a_pct_understory=hsi_instance.pct_understory,
            v3b_pct_midstory=hsi_instance.pct_midstory,
            v4a_flood_duration=hsi_instance.flood_duration,
            v4b_flow_exchange=hsi_instance.flow_exchange,
            v5_forested_connectivity_cat=hsi_instance.forested_connectivity_cat,
            dem_480=hsi_instance.dem_480,
            hydro_domain_480=hsi_instance.hydro_domain_480,
            pct_blh_cover=hsi_instance.pct_blh,
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

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _create_template_array(self) -> np.ndarray:
        """Create an array from a template all valid pixels are 999.0, and
        NaN from the input are persisted.
        """
        # Bottomland Hardwood Wetland Value Assessment (BLH WVA) has depth related vars, and is
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
        self._logger = logging.getLogger("BottomlandHardwoodHSI")
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

    def blh_cover_mask(self, si_array: np.ndarray) -> np.ndarray:
        """To apply the BLH WVA, at least 40% BLH cover (Zone 3 to 5)
        has to be present. This applies a mask to each SI array where
        the %BLH cover is < 40%. These areas are given an SI = NaN.
        """
        if self.pct_blh_cover is not None:
            blh_mask = (
                (self.pct_blh_cover < 40) 
                & (~np.isnan(self.pct_blh_cover))
                & (~np.isnan(self.pct_blh_cover))
                & (~np.isnan(self.dem_480)) # include domain
            )  
            si_array[blh_mask] = np.nan

        return si_array

    def calculate_si_1(self) -> np.ndarray:
        """Tree Species Association"""
        self._logger.info("Running SI 1")
        si_1 = self.template.copy()

        if (
            self.v1a_pct_overstory_w_mast is None
            or self.v1b_pct_overstory_w_soft_mast is None
            or self.v1c_pct_overstory_w_hard_mast is None
        ):
            self._logger.info(
                "Overstory with mast values not provided. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else:
            # class 1
            mask_1 = (self.v1a_pct_overstory_w_mast < 25) | (
                (self.v1b_pct_overstory_w_soft_mast > 50)
                & (self.v1c_pct_overstory_w_hard_mast == 0)
            )
            si_1[mask_1] = 0.2

            # class 2
            mask_2 = (
                (self.v1a_pct_overstory_w_mast >= 25)
                & (self.v1a_pct_overstory_w_mast <= 50)
                & (self.v1c_pct_overstory_w_hard_mast < 10)
            )
            si_1[mask_2] = 0.4

            # class 3
            mask_3 = (
                (self.v1a_pct_overstory_w_mast >= 25)
                & (self.v1a_pct_overstory_w_mast <= 50)
                & (self.v1c_pct_overstory_w_hard_mast >= 10)
            )
            si_1[mask_3] = 0.6

            # class 4
            mask_4 = (self.v1a_pct_overstory_w_mast > 50) & (
                self.v1c_pct_overstory_w_hard_mast < 20
            )
            si_1[mask_4] = 0.8

            # class 5
            mask_5 = (self.v1a_pct_overstory_w_mast > 50) & (
                self.v1c_pct_overstory_w_hard_mast >= 20
            )
            si_1[mask_5] = 1

        si_1 = self.blh_cover_mask(si_1)

        if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_1)

    def calculate_si_2(self) -> np.ndarray:
        """Stand Maturity"""
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()

        if self.v2_stand_maturity is None:
            self._logger.info(
                "Stand maturity (tree age) is not provided. Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            # condition 1
            mask_1 = self.v2_stand_maturity == 0
            si_2[mask_1] = 0

            # condition 2
            mask_2 = (self.v2_stand_maturity > 0) & (
                self.v2_stand_maturity <= 3
            )
            si_2[mask_2] = 0.0033 * self.v2_stand_maturity[mask_2]

            # condition 3
            mask_3 = (self.v2_stand_maturity > 3) & (
                self.v2_stand_maturity <= 7
            )
            si_2[mask_3] = (0.01 * self.v2_stand_maturity[mask_3]) - 0.02

            # condition 4
            mask_4 = (self.v2_stand_maturity > 7) & (
                self.v2_stand_maturity <= 10
            )
            si_2[mask_4] = (0.017 * self.v2_stand_maturity[mask_4]) - 0.07

            # condition 5
            mask_5 = (self.v2_stand_maturity > 10) & (
                self.v2_stand_maturity <= 20
            )
            si_2[mask_5] = (0.02 * self.v2_stand_maturity[mask_5]) - 0.1

            # condition 6
            mask_6 = (self.v2_stand_maturity > 20) & (
                self.v2_stand_maturity <= 30
            )
            si_2[mask_6] = (0.03 * self.v2_stand_maturity[mask_6]) - 0.3

            # condition 7
            mask_7 = (self.v2_stand_maturity > 30) & (
                self.v2_stand_maturity <= 50
            )
            si_2[mask_7] = 0.02 * self.v2_stand_maturity[mask_7]

            # condition 8
            mask_8 = self.v2_stand_maturity > 50
            si_2[mask_8] = 1

        si_2 = self.blh_cover_mask(si_2)

        if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_2)

    def calculate_si_3(self) -> np.ndarray:
        """Understory/Midstory"""

        # Understory SI
        self._logger.info("Running Understory SI")
        u_si = self.template.copy()

        if self.v3a_pct_understory is None:
            self._logger.info(
                "Understory data not provided. Setting index to 1."
            )
            u_si[~np.isnan(u_si)] = 1

        else:
            # understory condition 1
            mask_u1 = self.v3a_pct_understory == 0
            u_si[mask_u1] = 0.1

            # understory condition 2
            mask_u2 = (self.v3a_pct_understory > 0) & (
                self.v3a_pct_understory <= 30
            )
            u_si[mask_u2] = (0.03 * self.v3a_pct_understory[mask_u2]) + 0.1

            # understory condition 3
            mask_u3 = (self.v3a_pct_understory > 30) & (
                self.v3a_pct_understory <= 60
            )
            u_si[mask_u3] = 1

            # understory condition 4
            mask_u4 = self.v3a_pct_understory > 60
            u_si[mask_u4] = (-0.01 * self.v3a_pct_understory[mask_u4]) + 1.6

        # Midstory SI
        self._logger.info("Running Midstory SI")
        m_si = self.template.copy()

        if self.v3b_pct_midstory is None:
            self._logger.info(
                "Midstory data not provided. Setting index to 1."
            )
            m_si[~np.isnan(m_si)] = 1

        else:
            # midstory condition 1
            mask_m1 = self.v3b_pct_midstory == 0
            m_si[mask_m1] = 0.1

            # midstory condition 2
            mask_m2 = (self.v3b_pct_midstory > 0) & (
                self.v3b_pct_midstory <= 20
            )
            m_si[mask_m2] = (0.045 * self.v3b_pct_midstory[mask_m2]) + 0.1

            # midstory condition 3
            mask_m3 = (self.v3b_pct_midstory > 20) & (
                self.v3b_pct_midstory <= 50
            )
            m_si[mask_m3] = 1

            # midstory condition 4
            mask_m4 = self.v3b_pct_midstory > 50
            m_si[mask_m4] = (-0.01 * self.v3b_pct_midstory[mask_m4]) + 1.5

        # Combined SI 3
        self._logger.info("Running SI 3")
        si_3 = self.template.copy()
        si_3 = (u_si + m_si) / 2

        si_3 = self.blh_cover_mask(si_3)

        if np.any(np.isclose(si_3, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_3)

    def calculate_si_4(self) -> np.ndarray:
        """Hydrology"""
        self._logger.info("Running SI 4")
        si_4 = self.template.copy()

        if self.v4a_flood_duration is None or self.v4b_flow_exchange is None:
            self._logger.info(
                "Flood duration or flow exchange data is not provided. Setting index to 1."
            )
            si_4[~np.isnan(si_4)] = 1

        else:
            # self.v4a_flood_duration and self.v4b_flow_exchange are NumPy arrays of strings
            # condition 1
            mask_1 = (self.v4a_flood_duration == "Temporary") & (
                self.v4b_flow_exchange == "High"
            )
            si_4[mask_1] = 1

            # condition 2
            mask_2 = (self.v4a_flood_duration == "Seasonal") & (
                self.v4b_flow_exchange == "High"
            )
            si_4[mask_2] = 0.85

            # condition 3
            mask_3 = (self.v4a_flood_duration == "Semi-Permanent") & (
                self.v4b_flow_exchange == "High"
            )
            si_4[mask_3] = 0.75

            # condition 4
            mask_4 = (self.v4a_flood_duration == "No Flooding") & (
                self.v4b_flow_exchange == "High"
            )
            si_4[mask_4] = 0.65

            # condition 5
            mask_5 = (self.v4a_flood_duration == "Temporary") & (
                self.v4b_flow_exchange == "None"
            )
            si_4[mask_5] = 0.5

            # condition 6
            mask_6 = (self.v4a_flood_duration == "Seasonal") & (
                self.v4b_flow_exchange == "None"
            )
            si_4[mask_6] = 0.4

            # condition 7
            mask_7 = (self.v4a_flood_duration == "Semi-Permanent") & (
                self.v4b_flow_exchange == "None"
            )
            si_4[mask_7] = 0.25

            # condition 8
            mask_8 = (self.v4a_flood_duration == "No Flooding") & (
                self.v4b_flow_exchange == "None"
            )
            si_4[mask_8] = 0.1

        si_4 = self.blh_cover_mask(si_4)

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
            # extra mask for ouput of 0 (not forest) in forested
            # connectivity, this becomes NaN
            mask_0 = self.v5_forested_connectivity_cat == 0
            si_5[mask_0] = np.nan

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

        si_5 = self.blh_cover_mask(si_5)

        if np.any(np.isclose(si_5, 999.0, atol=1e-5)):
            raise ValueError("Unhandled condition in SI logic!")

        return self.clip_array(si_5)

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
        hsi[mask_1] = (
            (self.si_2[mask_1] ** 4) * (self.si_4[mask_1] ** 2)
        ) ** (1 / 6)

        # condition 2 (tree age >= 7 and v3_understory/midstory data is available)
        mask_2 = self.v2_stand_maturity >= 7
        hsi[mask_2] = (
            (self.si_1[mask_2] ** 4)
            * (self.si_2[mask_2] ** 4)
            * (self.si_3[mask_2] ** 2)
            * (self.si_4[mask_2] ** 2)
            * (self.si_5[mask_2])
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
        # The blh_mask is already applied via the individual SIs.
        masked_hsi = np.where(np.isnan(self.dem_480), np.nan, hsi)

        return masked_hsi
