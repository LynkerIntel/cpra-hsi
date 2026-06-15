# A module for non-hsi metrics needed in the DST. This module will be called during the HSI post_process routine.
# Initially, CW will be handling most metrics so this will stay as a helper module, supplying arrays in the correct
# aggregations and subsets for CW's metrics code. Eventually, we may take over the bulk of the metrics code, in which
# case this may be expanded into a class that runs independtly of HSI. Metrics should only be calculated here if they
# are independent of the HSI models (i.e. sediment flux)--otherwise they should be calculated and serialized during
# the HSI model run loop.

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
