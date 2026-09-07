# A module for non-hsi metrics needed in the DST. This module will be called during the HSI post_process routine.
# Initially, CW will be handling most metrics so this will stay as a helper module, supplying arrays in the correct
# aggregations and subsets for CW's metrics code. Eventually, we may take over the bulk of the metrics code, in which
# case this may be expanded into a class that runs independtly of HSI. Metrics should only be calculated here if they
# are independent of the HSI models (i.e. sediment flux)--otherwise they should be calculated and serialized during
# the HSI model run loop.

from typing import List
import numpy as np
import xarray as xr


def mask_sedflux_by_polygons(
    polygons_path: str,
    sedflux_paths: List[str],
    years: List[int],
    output_path: str,
) -> None:
    """Mask per-WY SEDFLUX data by subobjective polygon groups.

    STUB. Mirrors the logic in cpra-metrics/delineate-subobjective-zones.py:
    rasterize polygons grouped by ``PolyName`` onto the SEDFLUX grid, apply
    each mask to the SEDFLUX data for each water year, and stack into a
    ``(time, zone, y, x)`` Dataset written to ``output_path``.

    Parameters
    ----------
    polygons_path : str
        Path to the subobjective polygon shapefile. Must have ``PolyName``,
        ``Subobjecti``, and ``Objective`` attribute columns.
    sedflux_paths : list[str]
        Per-water-year SEDFLUX zarr paths, in the same order as ``years``.
    years : list[int]
        Simulation water years, used as the ``time`` coordinate.
    output_path : str
        Destination NetCDF path.
    """
    raise NotImplementedError(
        "mask_sedflux_by_polygons is a stub — port logic from "
        "cpra-metrics/delineate-subobjective-zones.py"
    )


def get_water_quality_metric(
    ds: xr.DataArray | xr.Dataset | None,
) -> np.ndarray | None:
    """
    Generate the annual dissolved oxygen metric for a single water year.

    Reduces one water year of daily dissolved oxygen to the July-September
    minimum of the 21-day rolling mean. A water year (Oct 1 - Sep 30) fully
    contains its own Jun-Sep, so the trailing 21-day window at the start of
    July never reaches outside the water year, and the metric can be built up
    one timestep at a time alongside the rest of the model variables.

    Parameters
    ----------
    ds : xr.DataArray | xr.Dataset | None
        Daily (time, y, x) dissolved oxygen for one water year. ``None`` when
        no dissolved oxygen input is configured, in which case ``None`` is
        returned.

    Returns
    -------
    np.ndarray or None
        The (y, x) July-September minimum of the 21-day rolling mean.
    """
    if ds is None:
        return None

    if isinstance(ds, xr.Dataset):
        ds = ds["dissolved_oxygen"]

    # subset first to reduce memory pressure,
    # but keep JAS + a June lookback buffer for the trailing 21-day window
    ds_pre = ds.sel(time=ds["time"].dt.month.isin([6, 7, 8, 9]))

    ds_rolled = ds_pre.rolling(time=21, min_periods=11).mean()
    ds_jas = ds_rolled.sel(time=ds_rolled["time"].dt.month.isin([7, 8, 9]))
    return ds_jas.min(dim="time", skipna=True).to_numpy()
