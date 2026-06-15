"""
Standalone script to generate daily dissolved oxygen predictions from
an XGBoost model trained on station data.

Inputs (zarr):
    - Water temperature (daily, 60m)
    - Water depth / stage (daily, 60m)
    - Velocity (daily, 60m)
    - DEM GeoTIFF (60m)

Output:
    - NetCDF with daily dissolved oxygen at 60m resolution

Usage:
    python run_dissolved_oxygen.py \\
        --data-dir /Users/dillonragar/data/cpra \\
        --group G400 --wy 22 --slr 328 \\
        --input-version V2 --output-version V3
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401 — registers rio accessor
from odc.geo.xr import write_cog
from xgboost import XGBRegressor

MODEL_PATH = "/Users/dillonragar/data/cpra/ml_out/xgb_dissolved_oxygen.json"


def drop_leap_days(ds: xr.Dataset) -> xr.Dataset:
    """Remove Feb 29 timesteps from a dataset."""
    dt = ds["time"].dt
    mask = ~((dt.month == 2) & (dt.day == 29))
    return ds.sel(time=mask)


def load_zarr_with_crs(path: str) -> xr.Dataset:
    """Open a zarr store and write CRS from its metadata."""
    ds = xr.open_zarr(path)
    crs_wkt = ds["crs"].attrs.get("crs_wkt")
    ds = ds.rio.write_crs(crs_wkt)
    return ds


def reproject_match(ds, dem: xr.DataArray):
    """Reproject dataset to match DEM grid if needed."""
    if ds.rio.crs == dem.rio.crs and ds.rio.bounds() == dem.rio.bounds():
        return ds
    return ds.rio.reproject_match(dem)


def compute_depth(stage_ds: xr.Dataset, dem: xr.DataArray) -> xr.DataArray:
    """Convert stage (WSE) to water depth by subtracting DEM."""
    stage_var = None
    for name in ["Band1", "waterlevel", "water_level", "stage"]:
        if name in stage_ds:
            stage_var = name
            break
    if stage_var is None:
        raise ValueError(
            f"Cannot find stage variable in dataset. "
            f"Available: {list(stage_ds.data_vars)}"
        )

    depth = stage_ds[stage_var] - dem
    depth = depth.where(~np.isnan(depth), 0.0)
    depth = depth.clip(min=0)
    depth.name = "height"
    return depth


def save_daily_cogs(
    ds: xr.Dataset,
    output_dir: str,
    overwrite: bool = False,
):
    """Save each variable and timestep as a Cloud Optimized GeoTIFF.

    Mirrors VegProcessor.utils.save_variables_as_cogs: reproject to Web
    Mercator, label files by water-year DOY, and write with
    odc.geo ``write_cog``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to export. Must have a ``time`` dimension.
    output_dir : str
        Root directory for the COG output.
    overwrite : bool
        If True, overwrite existing COGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for var_name in ds.data_vars:
        var = ds[var_name]
        var_dir = os.path.join(output_dir, var_name)
        os.makedirs(var_dir, exist_ok=True)
        print(
            f"\nProcessing variable: {var_name} (reprojecting per-slice to EPSG:3857)"
        )

        for t_idx in range(len(ds.time.values)):
            ts = pd.Timestamp(ds.time.values[t_idx])
            # Water year starts Oct 1: Oct-Dec belong to next calendar year's WY
            wy = ts.year + 1 if ts.month >= 10 else ts.year
            wy_start = pd.Timestamp(year=wy - 1, month=10, day=1)
            doy = (ts.normalize() - wy_start).days + 1
            file_label = f"WY{wy}_DOY_{doy:03d}"

            da_slice = var.isel(time=t_idx).squeeze(drop=True)
            da_slice = da_slice.rio.reproject("EPSG:3857")
            min_value = float(da_slice.min().values)
            max_value = float(da_slice.max().values)
            units = da_slice.attrs.get("units", "")

            output_path = os.path.join(var_dir, f"{var_name}_{file_label}.tif")
            if not overwrite and os.path.exists(output_path):
                continue
            print(
                f"  writing: {output_path} "
                f"(min: {min_value:.3f}, max: {max_value:.3f})"
            )
            write_cog(
                da_slice,
                fname=output_path,
                overwrite=overwrite,
                blocksize=512,
                compress="deflate",
                tags={
                    "MIN_VALUE": min_value,
                    "MAX_VALUE": max_value,
                    "UNIT": units,
                },
            )
            del da_slice
            gc.collect()
        del var
        gc.collect()
    print("Done exporting COGs.")


def predict_do(
    data_dir: str,
    group: str,
    wy: str,
    slr: str,
    input_version: str,
    output_version: str,
    predictors_out: bool,
):
    """Run daily dissolved oxygen prediction and save to NetCDF."""
    temperature_path = f"{data_dir}/AMP_INPUT/AMP_D3D_WY{wy}_{slr}_FX_99_99_DLY_{group}_AB_O_WTEMP_{input_version}.zarr"
    depth_path = f"{data_dir}/AMP_INPUT/AMP_D3D_WY{wy}_{slr}_FX_99_99_DLY_{group}_AB_O_STAGE_{input_version}.zarr"
    velocity_path = f"{data_dir}/AMP_INPUT/AMP_D3D_WY{wy}_{slr}_FX_99_99_DLY_{group}_AB_O_VELOCITY_{input_version}.zarr"
    dem_path = f"{data_dir}/60m_dem_1280_3200_padded.tif"
    domain_path = f"{data_dir}/D3D_model_domain.tif"
    output_dir = f"{data_dir}/data_staging/do"
    output_stem = (
        f"AMP_XGB_WY{wy}_{slr}_FX_99_99_DLY_{group}_AB_O_DO_{output_version}"
    )
    output_nc_path = f"{output_dir}/{output_stem}.nc"
    output_cog_dir = f"{output_dir}/{output_stem}_cogs"

    # Load DEM
    print("Loading DEM...")
    dem = xr.open_dataarray(dem_path)
    dem = dem.squeeze(drop="band")
    dem = dem.rio.write_crs("EPSG:6344")

    # Load hydro domain mask (same as VegTransition / HSI)
    print("Loading hydro domain mask...")
    domain = xr.open_dataarray(domain_path)
    domain = domain.squeeze(drop="band")
    domain = domain.rio.write_crs("EPSG:6344")
    domain = domain.rio.reproject_match(dem)
    domain_mask = ~np.isnan(domain.values)  # True = valid domain pixel

    # Load XGBoost model
    print(f"Loading DO model from {MODEL_PATH}...")
    xgb = XGBRegressor()
    xgb.load_model(MODEL_PATH)

    # Load temperature eagerly (avoid dask scheduler overhead)
    print(f"Loading temperature from {temperature_path}...")
    temp_ds = load_zarr_with_crs(temperature_path)
    if "wtemp" in temp_ds.data_vars:
        temp_ds = temp_ds.rename({"wtemp": "temperature"})
    temp_ds = reproject_match(temp_ds, dem)
    temp_ds = drop_leap_days(temp_ds)
    temp = temp_ds["temperature"].load()  # (time, y, x) — into memory

    # Load depth (stage -> depth via DEM subtraction), eagerly
    print(f"Loading stage/depth from {depth_path}...")
    stage_ds = load_zarr_with_crs(depth_path)
    stage_ds = reproject_match(stage_ds, dem)
    stage_ds = drop_leap_days(stage_ds)
    depth = compute_depth(stage_ds, dem).load()  # (time, y, x) — into memory

    # Load velocity (daily timesteps)
    print(f"Loading velocity from {velocity_path}...")
    vel_ds = load_zarr_with_crs(velocity_path)
    vel_ds = reproject_match(vel_ds, dem)
    vel_ds = drop_leap_days(vel_ds)
    if vel_ds["velocity"].sizes.get("time", 1) <= 1:
        raise ValueError(
            f"Velocity data must have daily timesteps, "
            f"but only has {vel_ds['velocity'].sizes.get('time', 0)} timestep(s). "
            f"Source: {velocity_path}"
        )

    # Align all inputs to common timestamps
    common_times = np.intersect1d(temp.time.values, depth.time.values)
    common_times = np.intersect1d(common_times, vel_ds["velocity"].time.values)
    print(f"Common timesteps across all inputs: {len(common_times)}")
    temp_ds = temp_ds.sel(time=common_times)
    temp = temp_ds["temperature"].load()
    depth = depth.sel(time=common_times)
    vel_ds = vel_ds.sel(time=common_times)
    vel_vals = vel_ds["velocity"].load().values  # (time, y, x)

    # Pre-extract numpy arrays and time metadata
    temp_vals = temp.values  # (time, y, x)
    depth_vals = depth.values  # (time, y, x)
    ny, nx = temp_vals.shape[1], temp_vals.shape[2]
    n_pixels = ny * nx

    print("Running DO prediction...")
    times = temp.time.values
    n_days = len(times)
    do_arr = np.empty((n_days, ny, nx), dtype=np.float32)
    t_start = time.time()

    for i in range(n_days):
        t = times[i]
        ts = pd.Timestamp(t)
        month = ts.month
        doy = ts.dayofyear

        vel_flat = vel_vals[i].ravel()

        features = np.column_stack(
            [
                temp_vals[i].ravel(),
                depth_vals[i].ravel(),
                vel_flat,
                np.full(n_pixels, month, dtype=np.float32),
                np.full(n_pixels, doy, dtype=np.float32),
            ]
        )
        pred = xgb.predict(features).reshape(ny, nx)
        pred[~domain_mask] = np.nan
        depth_mask = (depth_vals[i] > 0.1) & (depth_vals[i] < 4.0)
        pred[~depth_mask] = np.nan
        do_arr[i] = pred

        if (i + 1) % 10 == 0 or (i + 1) == n_days:
            elapsed = time.time() - t_start
            per_day = elapsed / (i + 1)
            remaining = per_day * (n_days - i - 1)
            date_str = str(t)[:10]
            print(
                f"  [{i + 1}/{n_days}] {date_str} — "
                f"{elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining"
            )

    # Build xarray DataArray from numpy result.
    do_da = xr.DataArray(
        do_arr,
        dims=("time", "y", "x"),
        coords={"time": times, "y": temp.y.values, "x": temp.x.values},
    )
    do_da.attrs = {
        "units": "mg/L",
        "long_name": "dissolved oxygen",
        "description": "Daily dissolved oxygen predicted by XGBoost model",
        "grid_mapping": "spatial_ref",
    }
    do_ds = xr.Dataset({"dissolved_oxygen": do_da})

    # Attach CF-compliant coordinate metadata so ArcGIS recognizes the CRS
    do_ds["x"].attrs.update(
        {
            "units": "m",
            "long_name": "Easting",
            "standard_name": "projection_x_coordinate",
        }
    )
    do_ds["y"].attrs.update(
        {
            "units": "m",
            "long_name": "Northing",
            "standard_name": "projection_y_coordinate",
        }
    )
    do_ds["time"].attrs.update(
        {
            "long_name": "time",
            "standard_name": "time",
        }
    )

    do_ds = do_ds.rio.write_crs("EPSG:6344")

    # Write NetCDF
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing NetCDF to {output_nc_path}...")
    do_ds.to_netcdf(
        output_nc_path,
        encoding={
            "dissolved_oxygen": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "grid_mapping": "spatial_ref",
            },
            "time": {
                "units": "days since 1850-01-01T00:00:00",
                "calendar": "gregorian",
                "dtype": "float64",
            },
        },
    )

    # Write daily COGs (DO output)
    print(f"Writing daily COGs to {output_cog_dir}...")
    save_daily_cogs(do_ds, output_cog_dir, overwrite=True)

    # Free DO intermediates before the input-COG loop to reduce memory pressure
    del do_arr, do_da, do_ds, temp_vals, depth_vals
    gc.collect()

    # Write daily COGs for input variables (QAQC)
    if predictors_out:
        input_cog_dir = os.path.join(output_dir, f"{output_stem}_inputs_cogs")
        print(f"Writing input COGs to {input_cog_dir}...")
        inputs_ds = xr.Dataset(
            {
                "temperature": temp,
                "water_depth": depth,
                "velocity": vel_ds["velocity"],
            }
        )
        inputs_ds = inputs_ds.rio.write_crs("EPSG:6344")
        save_daily_cogs(inputs_ds, input_cog_dir, overwrite=True)
    else:
        print("Skipping predictor input COGs (--predictors-out not set).")
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate daily dissolved oxygen predictions from an XGBoost "
            "model trained on station data."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root data directory (e.g. /Users/dillonragar/data/cpra).",
    )
    parser.add_argument(
        "--group", required=True, help="Group code, e.g. G400."
    )
    parser.add_argument("--wy", required=True, help="Water year, e.g. 22.")
    parser.add_argument(
        "--slr", required=True, help="SLR scenario code, e.g. 328."
    )
    parser.add_argument(
        "--input-version",
        required=True,
        help="Input dataset version tag (e.g. V2).",
    )
    parser.add_argument(
        "--output-version",
        required=True,
        help="Output dataset version tag (e.g. V3).",
    )
    parser.add_argument(
        "--predictors-out",
        action="store_true",
        help="Also write daily COGs of the predictor inputs for QAQC.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    args = parse_args()

    completed = False
    try:
        predict_do(
            data_dir=args.data_dir,
            group=args.group,
            wy=args.wy,
            slr=args.slr,
            input_version=args.input_version,
            output_version=args.output_version,
            predictors_out=args.predictors_out,
        )
        completed = True
    finally:
        if not completed:
            banner = "!" * 60
            print(f"\n{banner}", file=sys.stderr)
            print(
                "WARNING: script exited BEFORE finishing. "
                "Outputs may be incomplete.",
                file=sys.stderr,
            )
            print(f"{banner}\n", file=sys.stderr)
            # Note: SIGKILL (e.g. OS OOM-kill) cannot be caught — if the
            # process dies with no traceback and no warning, suspect that.
