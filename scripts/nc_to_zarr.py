"""Convert NetCDF files to Zarr stores (1:1).

Every .nc file found recursively under the input directory is converted
to a .zarr store with the same name. The subfolder layout is mirrored
under the output directory (e.g. ``input/STAGE/file.nc`` →
``output/STAGE/file.zarr``). Optionally reprojects each dataset to match
the grid of a reference raster using ``rioxarray.reproject_match()``.

Usage:
    python scripts/nc_to_zarr.py /data/hydro/mike_stage/
    python scripts/nc_to_zarr.py /data/hydro/G900/            # recurses into subfolders
    python scripts/nc_to_zarr.py /data/hydro/mike_stage/ -o /data/zarr_stores/
    python scripts/nc_to_zarr.py /data/hydro/mike_stage/ --time-chunks 10
    python scripts/nc_to_zarr.py /data/hydro/mike_stage/ --match-raster /data/dem.tif
"""

import argparse
import re
import sys
from pathlib import Path

import cftime
import pandas as pd
import rioxarray  # dont remove
import xarray as xr


def _open_dataset_with_time_fix(nc_path: Path, **kwargs) -> xr.Dataset:
    """Open a NetCDF dataset, fixing the off-by-two error for epoch 0001-01-01.

    Files with time units like "days since 0001-01-01" are misdecoded by
    xarray's default CF decoding because the "standard" calendar switches
    from Julian to Gregorian in 1582, introducing a 2-day offset for that
    epoch.  Forcing ``proleptic_gregorian`` fixes this.

    Files with other epochs are decoded normally via ``xr.decode_cf``.
    """
    ds = xr.open_dataset(nc_path, decode_times=False, **kwargs)

    if "time" in ds and "units" in ds["time"].attrs:
        units = ds["time"].attrs["units"]
        if "0001-01-01" in units or "0001-1-1" in units:
            calendar = "proleptic_gregorian"
            print(f"  Epoch 0001-01-01 detected — decoding with {calendar} calendar")
            time_vals = ds["time"].values
            dates = cftime.num2date(time_vals, units, calendar=calendar)
            timestamps = pd.DatetimeIndex(
                [
                    pd.Timestamp(d.year, d.month, d.day, d.hour, d.minute, d.second)
                    for d in dates
                ]
            )
            ds["time"] = ("time", timestamps)
            # Clean up encoding attrs so downstream decode_cf doesn't re-decode
            for attr in ("units", "calendar"):
                ds["time"].attrs.pop(attr, None)
                ds["time"].encoding.pop(attr, None)
            return ds

    # Normal epoch — let xarray decode as usual
    return xr.decode_cf(ds)


def convert_file(
    nc_path: Path,
    output_path: Path,
    time_chunks: int = 1,
    match_raster: Path | None = None,
) -> Path:
    """Convert a single NetCDF file to a Zarr store.

    Parameters
    ----------
    nc_path : Path
        Input .nc file.
    output_path : Path
        Destination .zarr path.
    time_chunks : int
        Chunk size along the time dimension.
    match_raster : Path, optional
        Reference raster whose grid (CRS, resolution, bounds) the
        dataset will be reprojected to via ``rio.reproject_match()``.

    Returns
    -------
    Path
        Path to the created Zarr store.
    """
    print(f"  Opening {nc_path.name}...")
    ds = _open_dataset_with_time_fix(nc_path, engine="h5netcdf", chunks="auto")

    # Force time coordinates to match the water year from the filename -------------
    if "time" in ds.dims and ds.sizes["time"] > 1:
        wy_match = re.search(r"_WY(\d{2})_", nc_path.name)
        if wy_match is None:
            raise ValueError(
                f"Cannot parse water year from filename: {nc_path.name}"
            )
        wy = int(f"20{wy_match.group(1)}")
        wy_start = pd.Timestamp(f"{wy - 1}-10-01")

        n = ds.sizes["time"]
        times = pd.DatetimeIndex(ds.time.values)
        print(f"  Time range in file: {times[0]} to {times[-1]} ({n} steps)")

        # Build the canonical 365-day water year (Oct 1 – Sep 30, no Feb 29)
        wy_end = pd.Timestamp(f"{wy}-09-30")
        expected = pd.date_range(wy_start, wy_end, freq="D")
        expected = expected[~((expected.month == 2) & (expected.day == 29))]
        # Trim or validate data to match the 365-day target
        ds = ds.isel(time=slice(0, len(expected)))
        ds = ds.assign_coords(time=("time", expected))
        print(
            f"  Forced time to WY{wy}: {expected[0].date()} – {expected[-1].date()} "
            f"({len(expected)} steps)"
        )

    # Sediment flux is cumulative – keep only the last timestep -----------------
    if (
        "cumulative_sediment_erosion_deposition" in ds.data_vars
        and "time" in ds.dims
    ):
        print(
            f"  cumulative_sediment_erosion_deposition detected — keeping only last timestep ({ds.time.values[-1]})"
        )
        ds = ds.isel(time=[-1])

    # handle varied CRS metadata locations between model files-----------------
    try:
        # D3D & MIKE: CRS from crs variable's crs_wkt attribute
        crs_wkt = ds["crs"].attrs.get("crs_wkt")
        ds = ds.rio.write_crs(crs_wkt)

    except Exception:
        try:
            # HEC-RAS: CRS from transverse_mercator variable's spatial_ref attribute
            crs_wkt = ds["transverse_mercator"].attrs.get("spatial_ref")
            ds = ds.rio.write_crs(crs_wkt)
        except Exception:
            # XGB / other: CRS from spatial_ref variable's crs_wkt attribute
            crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
            ds = ds.rio.write_crs(crs_wkt)

    # Normalize spatial dimension names to y/x
    _DIM_RENAME = {"Northings": "y", "Eastings": "x"}
    rename = {k: v for k, v in _DIM_RENAME.items() if k in ds.dims}
    if rename:
        print(f"  Renaming dimensions: {rename}")
        ds = ds.rename(rename)

    # Normalize variable names
    _VAR_RENAME = {"wtemp": "temperature", "waterlevel": "stage", "water_level": "stage"}
    var_rename = {k: v for k, v in _VAR_RENAME.items() if k in ds.data_vars}
    if var_rename:
        print(f"  Renaming variables: {var_rename}")
        ds = ds.rename(var_rename)

    ds = ds.chunk({"time": time_chunks})

    if match_raster is not None:
        ds = _reproject_match(ds, match_raster)

    print(f"  Writing {output_path.name}...")
    ds.to_zarr(output_path, mode="w")
    ds.close()

    return output_path


def _reproject_match(ds: xr.Dataset, match_raster: Path) -> xr.Dataset:
    """Reproject *ds* to match the grid of *match_raster*."""
    ds_match = xr.open_dataset(match_raster)
    da_match = ds_match.squeeze(drop="band_data").to_dataarray(dim="band")

    crs_match = ds.rio.crs == da_match.rio.crs
    bounds_match = ds.rio.bounds() == da_match.rio.bounds()
    res_match = ds.rio.resolution() == da_match.rio.resolution()

    print(
        f"  reproject_match — CRS match: {crs_match}, "
        f"bounds match: {bounds_match}, resolution match: {res_match}"
    )

    if crs_match and bounds_match and res_match:
        print("  reproject_match skipped: already matches reference raster")
        da_match.close()
        return ds

    print("  Reprojecting to match reference raster...")
    ds_reprojected = ds.rio.reproject_match(da_match)
    da_match.close()
    return ds_reprojected


def nc_to_zarr(
    input_dir: Path,
    output_dir: Path | None = None,
    time_chunks: int = 1,
    match_raster: Path | None = None,
) -> list[Path]:
    """Convert all .nc files under a directory to .zarr stores (recursive).

    Recurses into subfolders and mirrors the input subfolder structure in
    the output directory — e.g. ``input_dir/STAGE/foo.nc`` becomes
    ``output_dir/STAGE/foo.zarr``.

    Parameters
    ----------
    input_dir : Path
        Directory containing .nc files (may include nested subfolders).
    output_dir : Path, optional
        Output directory for .zarr stores. Defaults to
        ``{input_dir}_zarr`` (sibling directory).
    time_chunks : int
        Chunk size along the time dimension. Default 1.
    match_raster : Path, optional
        Reference raster whose grid (CRS, resolution, bounds) each
        dataset will be reprojected to via ``rio.reproject_match()``.

    Returns
    -------
    list of Path
        Paths to the created Zarr stores.
    """
    input_dir = Path(input_dir)
    nc_files = sorted(input_dir.rglob("*.nc"))
    if not nc_files:
        print(f"No .nc files found under {input_dir}")
        sys.exit(1)

    if output_dir is None:
        output_dir = input_dir.parent / (input_dir.name + "_zarr")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(nc_files)} NetCDF files under {input_dir}")
    for f in nc_files:
        rel = f.relative_to(input_dir)
        print(f"  {rel} ({f.stat().st_size / 1024**2:.0f} MB)")

    results = []
    for nc_file in nc_files:
        rel = nc_file.relative_to(input_dir)
        out = output_dir / rel.with_suffix(".zarr")
        out.parent.mkdir(parents=True, exist_ok=True)
        result = convert_file(nc_file, out, time_chunks, match_raster)
        results.append(result)

    print(f"\n{'=' * 60}")
    print(f"Converted {len(results)} file(s) to {output_dir}")
    for p in results:
        print(f"  {p.relative_to(output_dir)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert NetCDF files to Zarr stores (1:1)."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .nc files (recurses into subfolders)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for .zarr stores (default: {input_dir}_zarr)",
    )
    parser.add_argument(
        "--time-chunks",
        type=int,
        default=1,
        help="Chunk size along time dimension (default: 1)",
    )
    parser.add_argument(
        "--match-raster",
        type=Path,
        default=None,
        help="Reference raster to reproject each dataset to (CRS, resolution, bounds)",
    )
    args = parser.parse_args()

    nc_to_zarr(
        args.input_dir, args.output_dir, args.time_chunks, args.match_raster
    )


if __name__ == "__main__":
    main()
