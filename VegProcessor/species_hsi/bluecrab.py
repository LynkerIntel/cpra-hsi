from dataclasses import dataclass, field
import logging
from math import exp
import numpy as np
import pandas as pd


@dataclass
class BlueCrabHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    # gridded data as numpy arrays or None
    # init with None to be distinct from np.nan
    v1a_mean_annual_salinity: np.ndarray = None
    v1b_mean_annual_temperature: np.ndarray = None
    v2_pct_emergent_vegetation: np.ndarray = None
    v1c_bluecrab_lookup_table: pd.DataFrame = None

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create BlueCrabHSI instance from an HSI instance."""
        return cls(
            v1a_mean_annual_salinity=hsi_instance.mean_annual_salinity,
            v1b_mean_annual_temperature=hsi_instance.mean_annual_temperature,
            v2_pct_emergent_vegetation=hsi_instance.pct_vegetated,
            # TODO implement these variables/inputs in hsi.py
            v1c_bluecrab_lookup_table=hsi_instance.bluecrab_lookup_table,
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

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _setup_logger(self):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("BlueCrabHSI")
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

    def calculate_si_1(self) -> np.ndarray:
        """Mean salinity and water temperature from the entire year."""
        if self.v1a_mean_annual_salinity is None:
            self._logger.info("Mean annual salinity data not provided. Setting index to 1.")
            si_1 = np.ones(self._shape)

        else:
            # Setup ideal values for mean annual temperature (HEC-RAS)
            if self.v1b_mean_annual_temperature is None:
                # self._logger.info("Mean annual temperature data not provided. Setting index to 1.")
                # si_1 = np.ones(self._shape)
                self._logger.info("Mean annual temperature data not provided. Using ideal conditions of 18 degrees C.")
                self.v1b_mean_annual_temperature = np.full(self._shape, 18)

            # SI Logic
            self._logger.info("Running SI 1")

            def get_CPUE_value(sal_m: float, wtemp_m: float) -> float:
                """return column value for 'cpue_scaled' where 'sal_m' and 'wtemp_m' are equal to the inputs in the lookup table, returning 999.0 if not found"""
                # TODO check if we need to clamp the sal_m and wtemp_m values like the CPRA HSI code does (lines 294-295 in their HSI.py)
                sal_m = round(sal_m, 1)
                wtemp_m = round(wtemp_m, 1)
                sal_m_column = self.v1c_bluecrab_lookup_table["sal_m"]
                wtemp_m_column = self.v1c_bluecrab_lookup_table["wtemp_m"]

                # rudimentary lookup code - could definitely be improved
                if sal_m in sal_m_column.values and wtemp_m in wtemp_m_column.values:
                    return self.v1c_bluecrab_lookup_table.loc[
                        (sal_m_column == sal_m) & (wtemp_m_column == wtemp_m),
                        "cpue_scaled",
                    ].values[0]
                else:
                    return 999.0

            si_1 = np.vectorize(get_CPUE_value)(
                self.v1a_mean_annual_salinity, self.v1b_mean_annual_temperature
            )

            # Check for unhandled condition with tolerance
            if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")
        
        return si_1

    def calculate_si_2(self) -> np.ndarray:
        """Percent of cell that is covered by emergent vegetation."""
        if self.v2_pct_emergent_vegetation is None:
            self._logger.info("Pct emergent vegetation data not provided. Setting index to 1.")
            si_2 = np.ones(self._shape)

        else:
            # SI Logic
            self._logger.info("Running SI 2")
            # Create an array to store the results
            si_2 = np.full(self._shape, 999.0)

            # condition 1
            mask_1 = self.v2_pct_emergent_vegetation < 0.25
            si_2[mask_1] = (0.03 * self.v2_pct_emergent_vegetation[mask_1]) + 0.25

            # condition 2
            mask_2 = (self.v2_pct_emergent_vegetation >= 0.25) & (
                self.v2_pct_emergent_vegetation <= 0.8
            )
            si_2[mask_2] = 1.0

            # condition 3
            mask_3 = self.v2_pct_emergent_vegetation > 0.8
            si_2[mask_3] = 5.0 - (0.05 * self.v2_pct_emergent_vegetation[mask_3])

            # Check for unhandled condition with tolerance
            if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

        return si_2

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running BlueCrab final HSI.")
        for si_name, si_array in [
            ("SI 1", self.si_1),
            ("SI 2", self.si_2),
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
        hsi = (self.si_1 * self.si_2) ** (1 / 2)

        # Check the final HSI array for invalid values
        invalid_values_hsi = (hsi < 0) | (hsi > 1)
        if np.any(invalid_values_hsi):
            num_invalid_hsi = np.count_nonzero(invalid_values_hsi)
            self._logger.warning(
                "Final HSI contains %d values outside the range [0, 1].",
                num_invalid_hsi,
            )

        return hsi
