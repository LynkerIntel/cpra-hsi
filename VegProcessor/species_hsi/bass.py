from dataclasses import dataclass, field
import logging
import numpy as np


@dataclass
class BassHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    hydro_domain_flag: bool  # If True, all HSI SI arrays are masked to
    # hydro domain. If False, SI arrays relying only on veg type will maintain entire
    # veg type domain, which is a greate area then hydro domain.
    hydro_domain_480: np.ndarray = None
    dem_480: np.ndarray = None

    # gridded data as numpy arrays or None
    # init with None to be distinct from np.nan
    v1a_mean_annual_salinity: np.ndarray = None
    v1b_mean_annual_temperature: np.ndarray = None
    v2_pct_emergent_vegetation: np.ndarray = None

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

    @classmethod
    def from_hsi(cls, hsi_instance):
        """Create BassHSI instance from an HSI instance."""
        return cls(
            v1a_mean_annual_salinity=hsi_instance.mean_annual_salinity,
            v1b_mean_annual_temperature=hsi_instance.mean_annual_temperature,
            v2_pct_emergent_vegetation=hsi_instance.pct_vegetated,
            dem_480=hsi_instance.dem_480,
            hydro_domain_480=hsi_instance.hydro_domain_480,
        )

    def __post_init__(self):
        """Run class methods to get HSI after instance is created."""
        # Set up the logger
        self._setup_logger()
        self.template = self._create_template_array()

        # Determine the shape of the arrays
        # self._shape = self._determine_shape()

        # Calculate individual suitability indices
        self.si_1 = self.calculate_si_1()
        self.si_2 = self.calculate_si_2()

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _setup_logger(self):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("BassHSI")
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
        arr = np.where(np.isnan(self.hydro_domain_480), np.nan, 999.0)
        return arr

    # def _determine_shape(self) -> tuple:
    #     """Determine the shape of the environmental variable arrays."""
    #     # Iterate over instance attributes and return the shape of the first non None numpy array
    #     for name, value in vars(self).items():
    #         if value is not None and isinstance(value, np.ndarray):
    #             self._logger.info("Using attribute %s as shape for output: %s", name, value.shape)
    #             return value.shape

    def calculate_si_1(self) -> np.ndarray:
        """Mean salinity and water temperature from the entire year."""
        self._logger.info("Running SI 1")
        si_1 = self.template.copy()

        if self.v1a_mean_annual_salinity is None:
            self._logger.info(
                "Mean annual salinity data not provided. Setting index to 1."
            )
            si_1[~np.isnan(si_1)] = 1

        else:
            # Setup ideal values for mean annual temperature (HEC-RAS)
            if self.v1b_mean_annual_temperature is None:
                # self._logger.info("Mean annual temperature data not provided. Setting index to 1.")
                # si_1 = np.ones(self._shape)
                self._logger.info(
                    "Mean annual temperature data not provided. Using ideal conditions of 18 degrees C."
                )
                self.v1b_mean_annual_temperature = self.template.copy()
                self.v1b_mean_annual_temperature[
                    ~np.isnan(self.v1b_mean_annual_temperature)
                ] = 18

            # SI Logic
            S_si = (self.v1a_mean_annual_salinity - 0.84) / 1.84
            T_si = (self.v1b_mean_annual_temperature - 22.68) / 4.64
            S_si_2 = (
                (self.v1a_mean_annual_salinity * self.v1a_mean_annual_salinity)
                - 4.08
            ) / 24.91
            T_si_2 = (
                (
                    self.v1b_mean_annual_temperature
                    * self.v1b_mean_annual_temperature
                )
                - 535.99
            ) / 206.16

            si_1 = (
                np.exp(
                    2.50
                    - (0.25 * S_si)
                    + (0.30 * T_si)
                    + (0.04 * S_si_2)
                    - (0.33 * T_si_2)
                    - (0.05 * (S_si * T_si))
                )
                / 14.3
            )

            if np.any(np.isclose(si_1, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

            # if self.hydro_domain_flag:
            #     si_1 = np.where(~np.isnan(self.hydro_domain_480), si_1, np.nan)

        return si_1

    def calculate_si_2(self) -> np.ndarray:
        """Percent of cell that is covered by emergent vegetation."""
        self._logger.info("Running SI 2")
        si_2 = self.template.copy()

        if self.v2_pct_emergent_vegetation is None:
            self._logger.info(
                "Pct emergent vegetation data not provided. Setting index to 1."
            )
            si_2[~np.isnan(si_2)] = 1

        else:
            # condition 1
            mask_1 = self.v2_pct_emergent_vegetation < 20
            si_2[mask_1] = 0.01

            # condition 2
            mask_2 = (self.v2_pct_emergent_vegetation >= 20) & (
                self.v2_pct_emergent_vegetation < 30
            )
            si_2[mask_2] = (
                0.099 * self.v2_pct_emergent_vegetation[mask_2]
            ) - 1.997

            # condition 3
            mask_3 = (self.v2_pct_emergent_vegetation >= 30) & (
                self.v2_pct_emergent_vegetation < 50
            )
            si_2[mask_3] = 1.0

            # condition 4
            mask_4 = (self.v2_pct_emergent_vegetation >= 50) & (
                self.v2_pct_emergent_vegetation < 85
            )
            si_2[mask_4] = (
                -0.0283 * self.v2_pct_emergent_vegetation[mask_4]
            ) + 2.414

            # condition 5
            mask_5 = (self.v2_pct_emergent_vegetation >= 85) & (
                self.v2_pct_emergent_vegetation < 100
            )
            si_2[mask_5] = 0.01

            # condition 6
            mask_3 = self.v2_pct_emergent_vegetation == 100
            si_2[mask_3] = 0.0

            # Check for unhandled condition with tolerance
            if np.any(np.isclose(si_2, 999.0, atol=1e-5)):
                raise ValueError("Unhandled condition in SI logic!")

            # if self.hydro_domain_flag:
            #     si_2 = np.where(~np.isnan(self.hydro_domain_480), si_2, np.nan)

        return si_2

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        self._logger.info("Running Bass final HSI.")
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

        # subset final HSI array to vegetation domain (not hydrologic domain)
        # Masking: Set values in `mask` to NaN wherever `data` is NaN
        masked_hsi = np.where(np.isnan(self.dem_480), np.nan, hsi)

        return masked_hsi
