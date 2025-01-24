from dataclasses import dataclass, field
import numpy as np
import logging


@dataclass
class BaldEagleHSI:
    """
    This dataclass handles ingestion of processed inputs to the HSI Model. Class methods are
    used for each individual suitability index, and initialized as None for cases where
    the data for an index is not available. These indices will be set to ideal (1).

    Note: All input vars are two dimensional np.ndarray with x, y, dims. All suitability index math
    should use numpy operators instead of `math` to ensure vectorized computation.
    """

    # gridded data as numpy arrays or None
    # init with None to be distinct from np.nan

    # JG //TODO: should prob split this out? 
    v1_pct_cell_developed_or_upland: np.ndarray = None
    #v1a_pct_cell_developed_land: np.ndarray = None
    #v1b_pct_cell_upland: np.ndarray = None
    v2_pct_cell_flotant_marsh: np.ndarray = None
    v3_pct_cell_forested_wetland: np.ndarray = None
    v4_pct_cell_fresh_marsh: np.ndarray = None
    v5_pct_cell_intermediate_marsh: np.ndarray = None
    v6_pct_cell_open_water: np.ndarray = None

    # Species-specific parameters (example values)
    # for cases where static values are used

    # optimal_temperature: float = 20.0  # EXAMPLE
    # temperature_tolerance: float = 10.0  # EXAMPLE

    # Suitability indices (calculated)
    si_1: np.ndarray = field(init=False)
    si_2: np.ndarray = field(init=False)
    si_3: np.ndarray = field(init=False)
    si_4: np.ndarray = field(init=False)
    si_5: np.ndarray = field(init=False)
    si_6: np.ndarray = field(init=False)

    # Overall Habitat Suitability Index (HSI)
    hsi: np.ndarray = field(init=False)

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

        # Calculate overall suitability score with quality control
        self.hsi = self.calculate_overall_suitability()

    def _setup_logger(self):
        """Set up the logger for the class."""
        self._logger = logging.getLogger("BaldEagleHSI")
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
        """Percent of cell that is developed land or upland."""
        # Calculate for inital conditions and use for all time periods
        
        #if self.v1a_pct_cell_developed_land is None | self.v1b_pct_cell_upland is None:
        if self.v1_pct_cell_developed_or_upland is None:
            self._logger.info("Pct developed land or upland data not provided. Setting index to 1.")
            si_1 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 1")
            # Create an array to store the results
            si_1 = np.full(self._shape, 999)

            # condition 1 (if no dev'd land or upland in cell)
            # note: we could just =0 vs !=0
            #mask_1 = self.v1a_pct_cell_developed_land | self.v1b_pct_cell_upland <= 0.0
            mask_1 = self.v1_pct_cell_developed_or_upland <= 0.0
            si_1[mask_1] =  0.01

            # condition 2 (otherwise)
            mask_2 = self.v1_pct_cell_developed_or_upland > 0.0
            si_1[mask_2] = 0.408 + 0.142 * np.log(self.v1_pct_cell_developed_or_upland[mask_2])

            if 999 in si_1:
                raise ValueError("Unhandled condition in SI logic!")

        return si_1
    
    def calculate_si_2(self) -> np.ndarray:
        """Percent of cell that is flotant marsh."""
        # Calculate for inital conditions and use for all time periods
        
        if self.v2_pct_cell_flotant_marsh is None:
            self._logger.info("Pct flotant marsh data not provided. Setting index to 1.")
            si_2 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 2")
            # Create an array to store the results
            si_2 = np.full(self._shape, 999)

            # condition 1 (there is just one function here, so no need for mask)
            si_2 = (0.282 + 
                (0.047 * self.v2_pct_cell_flotant_marsh) - 
                (1.105 * np.exp(-3) * self.v2_pct_cell_flotant_marsh ** 2) +
                (1.101 * np.exp(-5) * self.v2_pct_cell_flotant_marsh ** 3) -
                (3.967 * np.exp(-8) * self.v2_pct_cell_flotant_marsh ** 4)
            )

            if 999 in si_2:
                raise ValueError("Unhandled condition in SI logic!")

        return si_2
    
    def calculate_si_3(self) -> np.ndarray:
        """Percent of cell that is covered by forested wetland."""
        
        if self.v3_pct_cell_forested_wetland is None:
            self._logger.info("Pct forested wetland data not provided. Setting index to 1.")
            si_3 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 3")
            # Create an array to store the results
            si_3 = np.full(self._shape, 999)

            # condition 1 (there is just one function here, so no need for mask)
            si_3 = (0.015 + 
                (0.048 * self.v3_pct_cell_forested_wetland) - 
                (1.178 * np.exp(-3) * self.v3_pct_cell_forested_wetland ** 2) +
                (1.366 * np.exp(-5) * self.v3_pct_cell_forested_wetland ** 3) -
                (5.673 * np.exp(-8) * self.v3_pct_cell_forested_wetland ** 4)
            )

            if 999 in si_3:
                raise ValueError("Unhandled condition in SI logic!")

        return si_3
    
    def calculate_si_4(self) -> np.ndarray:
        """Percent of cell that is covered by fresh marsh."""
        
        if self.v4_pct_cell_fresh_marsh is None:
            self._logger.info("Pct fresh marsh data not provided. Setting index to 1.")
            si_4 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 4")
            # Create an array to store the results
            si_4 = np.full(self._shape, 999)

            # condition 1 (there is just one function here, so no need for mask)
            si_4 = (0.370 + 
                (0.07 * self.v4_pct_cell_fresh_marsh) - 
                (2.655 * np.exp(-3) * self.v4_pct_cell_fresh_marsh ** 2) +
                (3.691 * np.exp(-5) * self.v4_pct_cell_fresh_marsh ** 3) -
                (1.701 * np.exp(-7) * self.v4_pct_cell_fresh_marsh ** 4)
            )

            if 999 in si_4:
                raise ValueError("Unhandled condition in SI logic!")

        return si_4
    
    def calculate_si_5(self) -> np.ndarray:
        """Percent of cell that is covered by intermediate marsh."""
        
        if self.v5_pct_cell_intermediate_marsh is None:
            self._logger.info("Pct intermediate marsh data not provided. Setting index to 1.")
            si_5 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 5")
            # Create an array to store the results
            si_5 = np.full(self._shape, 999)

            # condition 1 (there is just one function here, so no need for mask)
            si_5 = (0.263 - 
                (9.406 * np.exp(-3) * self.v5_pct_cell_intermediate_marsh) +
                (5.432 * np.exp(-4) * self.v5_pct_cell_intermediate_marsh ** 2) -
                (3.817 * np.exp(-6) * self.v5_pct_cell_intermediate_marsh ** 3)
            )

            if 999 in si_5:
                raise ValueError("Unhandled condition in SI logic!")

        return si_5
    
    def calculate_si_6(self) -> np.ndarray:
        """Percent of cell that is open water."""
        if self.v6_pct_cell_open_water is None:
            self._logger.info("Pct open water data not provided. Setting index to 1.")
            si_6 = np.ones(self._shape)

        else:
            self._logger.info("Running SI 1")
            # Create an array to store the results
            si_6 = np.full(self._shape, 999)

            # condition 1
            mask_1 = self.v6_pct_cell_open_water <= 0.0
            si_6[mask_1] = 0.01

            # condition 2 (AND)
            mask_2 = (self.v6_pct_cell_open_water > 0.0) & (self.v6_pct_cell_open_water <= 0.95)
            si_6[mask_2] = 0.985 - (0.105 * (self.v6_pct_cell_open_water ** -1))

            # condition 3
            mask_3 = self.v6_pct_cell_open_water > 0.95
            si_6[mask_3] = 0.0

            if 999 in si_6:
                raise ValueError("Unhandled condition in SI logic!")

        return si_6

    def calculate_overall_suitability(self) -> np.ndarray:
        """Combine individual suitability indices to compute the overall HSI with quality control."""
        hsi = ((self.si_1 ** 0.0104) *
               (self.si_2 ** 0.3715) *
               (self.si_3 ** 0.4743) *
               (self.si_4 ** 0.033) *
               (self.si_5 ** 0.0353) *
               (self.si_6 ** 0.0669)) ** 0.991

        # Quality control check: Ensure combined_score is between 0 and 1
        invalid_values = (hsi < 0) | (hsi > 1)
        if np.any(invalid_values):
            num_invalid = np.count_nonzero(invalid_values)
            self._logger.warning(
                "Combined suitability score has %d values outside [0,1]. Clipping values.",
                num_invalid,
            )
            # Clip the combined_score to ensure it's between 0 and 1
            # hsi = np.clip(hsi, 0, 1)

        return hsi
