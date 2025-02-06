import unittest
import numpy as np
import pandas as pd
import xarray as xr
import veg_logic


class TestZoneV(unittest.TestCase):

    def setUp(self):
        """Set up test data for Zone V logic tests."""
        # Create a small veg_type array with consistent dimensions
        self.veg_type = np.array(
            [
                [15, 15, 0],  # Zone V pixels (15) and one non-Zone V (0)
                [0, 15, 15],  # Mixed pixels
            ]
        )

        # Create time range
        self.time = pd.date_range("1999-03-01", "1999-09-30", freq="MS")  # 7 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        self.wse_mean = np.array(
            [
                [  # x=0 row
                    [0, 0, 0, 0, 0, 0, 0],  # y=0 column
                    [0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5],  # y=1 column
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # y=2 column
                ],
                [  # x=1 row
                    [-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5],  # y=0 column
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],  # y=1 column
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # y=2 column
                ],
            ]
        )

        # Create xarray Dataset with consistent spatial dimensions
        self.water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], self.wse_mean)},
            coords={
                "x": np.arange(self.veg_type.shape[0]),  # Match `veg_type` shape
                "y": np.arange(self.veg_type.shape[1]),  # Match `veg_type` shape
                "time": self.time,
            },
        )
        # Reorder to match WSE data
        self.water_depth = self.water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [15, 16, np.nan],
                [np.nan, 15, 15],
            ]
        )

        result = veg_logic.zone_v(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


class TestZoneIV(unittest.TestCase):

    def setUp(self):
        """Set up test data for Zone IV logic tests."""
        # Create a small veg_type array with consistent dimensions
        self.veg_type = np.array(
            [
                [16, 16, 0],  # Zone V pixels (15) and one non-Zone V (0)
                [0, 16, 16],  # Mixed pixels
            ]
        )

        # Create time range
        time = pd.date_range("1999-03-01", "1999-09-30", freq="MS")  # 7 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        # Shape of wse_mean should be (2, 3, 7) to match (x, y, time)
        wse_mean = np.array(
            [
                [  # y=1 row
                    [-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5],  # x=0 column, time=7
                    [0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5],  # x=1 column
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # x=2 column
                ],
                [  # y=0 row
                    [-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5],  # x y=0 column
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],  # x=1 column
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # x=2 column
                ],
            ]
        )

        # Create xarray Dataset with consistent spatial dimensions
        water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], wse_mean)},
            coords={
                "x": np.arange(self.veg_type.shape[0]),  # Match `veg_type` shape
                "y": np.arange(self.veg_type.shape[1]),  # Match `veg_type` shape
                "time": time,
            },
        )
        # reorder to match WSE data
        self.water_depth = water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [17, 17, np.nan],
                [np.nan, 15, 16],
            ]
        )

        result = veg_logic.zone_iv(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


class TestZoneIII(unittest.TestCase):

    def setUp(self):
        """Set up test data for Zone III logic tests."""
        # Create a small veg_type array with consistent dimensions
        self.veg_type = np.array(
            [
                [17, 17, 0],  # Zone V pixels (15) and one non-Zone V (0)
                [0, 17, 17],
            ]
        )

        # Create time range
        time = pd.date_range("1999-03-01", "1999-09-30", freq="MS")  # 7 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        # Shape of wse_mean should be (2, 3, 7) to match (x, y, time)
        wse_mean = np.array(
            [
                [  # y=1 row
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # top middle
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # NaN
                ],
                [  # y=0 row
                    [-0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # NaN
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # bottom middle
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        # Create xarray Dataset with consistent spatial dimensions
        water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], wse_mean)},
            coords={
                "x": np.arange(self.veg_type.shape[0]),  # Match `veg_type` shape
                "y": np.arange(self.veg_type.shape[1]),  # Match `veg_type` shape
                "time": time,
            },
        )
        # reorder to match WSE data
        self.water_depth = water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [17, 18, np.nan],
                [np.nan, 16, 17],
            ]
        )

        result = veg_logic.zone_iii(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


class TestZoneII(unittest.TestCase):

    def setUp(self):
        """Set up test data for Zone II logic tests."""
        # Create a small veg_type array with consistent dimensions
        self.veg_type = np.array(
            [
                [18, 18, 0],
                [0, 18, 18],
            ]
        )

        # Create time range

        time = pd.date_range("1999-10-01", "2000-09-30", freq="MS")  # 12 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        # Shape of wse_mean should be (2, 3, 7) to match (x, y, time)
        wse_mean = np.array(
            [
                [  # y=1 row
                    [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 01.5, 1.5],
                    [
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],  # top middle
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # NaN
                ],
                [  # y=0 row
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],  # NaN
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # low mid
                    [
                        0.05,
                        0.05,
                        0.05,
                        0.05,
                        0.05,
                        0.05,
                        0.05,
                        0.05,
                        0.05,
                        0.05,
                        0.05,
                        0.05,
                    ],
                ],
            ]
        )

        # Create xarray Dataset
        water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], wse_mean)},
            coords={
                "time": time,
                "x": np.arange(self.veg_type.shape[0]),
                "y": np.arange(self.veg_type.shape[1]),
            },
        )
        # reorder to match WSE data
        self.water_depth = water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [19, 18, np.nan],
                [np.nan, 17, 20],
            ]
        )

        result = veg_logic.zone_ii(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


class TestFreshShrub(unittest.TestCase):

    def setUp(self):
        """Set up test data for Fresh Shrub logic tests."""
        # Create a small veg_type array with consistent dimensions
        self.veg_type = np.array(
            [
                [19, 19, 0],
                [0, 19, 19],
            ]
        )

        # Create time range

        time = pd.date_range("1999-10-01", "2000-09-30", freq="MS")  # 12 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        # Shape of wse_mean should be (2, 3, 7) to match (x, y, time)
        wse_mean = np.array(
            [
                [  # y=1 row
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],  # top middle
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # NaN
                ],
                [  # y=0 row
                    [
                        0.0,
                        -0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],  # NaN
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # low mid
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ],
            ]
        )

        # Create xarray Dataset
        water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], wse_mean)},
            coords={
                "time": time,
                "x": np.arange(self.veg_type.shape[0]),
                "y": np.arange(self.veg_type.shape[1]),
            },
        )
        # reorder to match WSE data
        self.water_depth = water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [20, 18, np.nan],
                [np.nan, 19, 20],
            ]
        )

        result = veg_logic.fresh_shrub(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


class TestFreshMarsh(unittest.TestCase):

    def setUp(self):
        """Set up test data for Fresh Marsh logic tests."""
        # Create a small veg_type array with consistent dimensions
        self.veg_type = np.array(
            [
                [20, 20, 20],
                [20, 20, 20],
            ]
        )
        self.salinity = np.array(
            [
                [3, 1, 1],
                [0, 1, 1],
            ]
        )
        # Create time range

        time = pd.date_range("1999-10-01", "2000-09-30", freq="MS")  # 12 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        # Shape of wse_mean should be (2, 3, 7) to match (x, y, time)
        wse_mean = np.array(
            [
                [  # y=1 row
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5],
                    [
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],  # top middle
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # NaN
                ],
                [  # y=0 row
                    [
                        0.0,
                        -0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],  # NaN
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # low mid
                    [
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ],  # low right
                ],
            ]
        )
        # Create xarray Dataset
        water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], wse_mean)},
            coords={
                "time": time,
                "x": np.arange(self.veg_type.shape[0]),
                "y": np.arange(self.veg_type.shape[1]),
            },
        )
        # reorder to match WSE data
        self.water_depth = water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [21, 26, 26],
                [26, 19, 18],
            ]
        )

        result = veg_logic.fresh_marsh(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            salinity=self.salinity,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


class TestIntermediateMarsh(unittest.TestCase):

    def setUp(self):
        """Set up test data for Intermediate Marsh logic tests."""
        # Create a small veg_type array with consistent dimensions
        self.veg_type = np.array(
            [
                [21, 21, 20],
                [21, 21, 21],
            ]
        )

        self.salinity = np.array(
            [
                [6, 1, 1],
                [0, 0.1, 1],
            ]
        )
        # Create time range

        time = pd.date_range("1999-10-01", "2000-09-30", freq="MS")  # 12 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        # Shape of wse_mean should be (2, 3, 7) to match (x, y, time)
        wse_mean = np.array(
            [
                [  # y=1 row
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                    [
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                    ],  # top middle
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # NaN
                ],
                [  # y=0 row
                    [
                        0.0,
                        -0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],  # NaN
                    [
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],  # low mid
                    [
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ],  # low right
                ],
            ]
        )
        # Create xarray Dataset
        water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], wse_mean)},
            coords={
                "time": time,
                "x": np.arange(self.veg_type.shape[0]),
                "y": np.arange(self.veg_type.shape[1]),
            },
        )
        # reorder to match WSE data
        self.water_depth = water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [22, 21, np.nan],
                [26, 20, 26],
            ]
        )

        result = veg_logic.intermediate_marsh(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            salinity=self.salinity,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


class TestBrackishMarsh(unittest.TestCase):

    def setUp(self):
        """Set up test data for Brackish Marsh logic tests."""
        # Create a small veg_type array
        self.veg_type = np.array(
            [
                [22, 22, 0],
                [22, 22, 22],
            ]
        )

        self.salinity = np.array(
            [
                [13, 6, 1],
                [0, 0.1, 1],
            ]
        )
        # Create time range

        time = pd.date_range("1999-10-01", "2000-09-30", freq="MS")  # 12 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        # Shape of wse_mean should be (2, 3, 7) to match (x, y, time)
        wse_mean = np.array(
            [
                [  # y=1 row
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                    [
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],  # top middle
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # NaN
                ],
                [  # y=0 row
                    [
                        0.0,
                        -0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],  # NaN
                    [
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],  # low mid
                    [
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ],  # low right
                ],
            ]
        )
        # Create xarray Dataset
        water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], wse_mean)},
            coords={
                "time": time,
                "x": np.arange(self.veg_type.shape[0]),
                "y": np.arange(self.veg_type.shape[1]),
            },
        )
        # reorder to match WSE data
        self.water_depth = water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [23, 22, np.nan],
                [26, 21, 26],
            ]
        )

        result = veg_logic.brackish_marsh(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            salinity=self.salinity,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


class TestSalineMarsh(unittest.TestCase):

    def setUp(self):
        """Set up test data for Saline Marsh logic tests."""
        # Create a small veg_type array
        self.veg_type = np.array(
            [
                [23, 23, 0],
                [23, 23, 23],
            ]
        )

        self.salinity = np.array(
            [
                [13, 6, 1],
                [0, 0.1, 1],
            ]
        )
        # Create time range

        time = pd.date_range("1999-10-01", "2000-09-30", freq="MS")  # 12 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        # Shape of wse_mean should be (2, 3, 7) to match (x, y, time)
        wse_mean = np.array(
            [
                [  # y=1 row
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                    [
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],  # top middle
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # NaN
                ],
                [  # y=0 row
                    [
                        0.0,
                        -0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],  # NaN
                    [
                        np.nan,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],  # low mid
                    [
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ],  # low right
                ],
            ]
        )
        # Create xarray Dataset
        water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], wse_mean)},
            coords={
                "time": time,
                "x": np.arange(self.veg_type.shape[0]),
                "y": np.arange(self.veg_type.shape[1]),
            },
        )
        # reorder to match WSE data
        self.water_depth = water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [23, 22, np.nan],
                [26, np.nan, 26],
            ]
        )

        result = veg_logic.saline_marsh(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            salinity=self.salinity,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


class TestWater(unittest.TestCase):

    def setUp(self):
        """Set up test data for Water logic tests."""
        # Create a small veg_type array
        self.veg_type = np.array(
            [
                [26, 26, 0],
                [26, 26, 26],
            ]
        )

        self.salinity = np.array(
            [
                [1, 4, 1],
                [9, 14, 1],
            ]
        )

        time = pd.date_range("1999-10-01", "2000-09-30", freq="MS")  # 12 time steps

        # Ensure wse_mean matches the spatial dimensions of veg_type
        # Shape of wse_mean should be (2, 3, 7) to match (x, y, time)
        wse_mean = np.array(
            [
                [  # y=1 row
                    [
                        0.01,
                        0.01,
                        0.01,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.08,
                    ],
                    [
                        0.01,
                        0.01,
                        0.01,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.08,
                    ],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # NaN
                ],
                [  # y=0 row
                    [
                        0.01,
                        0.01,
                        0.01,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.08,
                    ],
                    [
                        0.01,
                        0.01,
                        0.01,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.02,
                        0.08,
                    ],
                    [
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ],  # low right
                ],
            ]
        )

        # Create xarray Dataset
        water_depth = xr.Dataset(
            {"WSE_MEAN": (["x", "y", "time"], wse_mean)},
            coords={
                "time": time,
                "x": np.arange(self.veg_type.shape[0]),
                "y": np.arange(self.veg_type.shape[1]),
            },
        )
        # reorder to match WSE data
        self.water_depth = water_depth.transpose("time", "x", "y")

    def test_transitions(self):
        """Test that the shape of veg_type matches the first two dimensions of wse_mean."""
        correct_result = np.array(
            [
                [20, 21, np.nan],
                [22, 23, 26],
            ]
        )

        result = veg_logic.water(
            veg_type=self.veg_type,
            water_depth=self.water_depth,
            salinity=self.salinity,
            timestep_output_dir="~/data/tmp/scratch/",
        )["veg_type"]

        np.testing.assert_array_equal(
            correct_result,
            result,
            "Result does not match known values.",
        )


if __name__ == "__main__":
    unittest.main()
