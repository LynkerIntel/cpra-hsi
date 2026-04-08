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
    python predict_dissolved_oxygen.py
"""

import os
import re

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401 — registers rio accessor
from odc.geo.xr import write_cog
from xgboost import XGBRegressor

# =========================================================
# PATHS — edit these before running
# =========================================================
DATA_DIR = "/Users/dillonragar/data/cpra"

TEMPERATURE_PATH = f"{DATA_DIR}/AMP_D3D_WTEMP/AMP_D3D_WY22_000_FX_99_99_DLY_G900_AB_O_WTEMP_V1.zarr"
DEPTH_PATH = f"{DATA_DIR}/AMP_D3D_STAGE/AMP_D3D_WY22_000_FX_99_99_DLY_G900_AB_O_STAGE_V1.zarr"
VELOCITY_PATH = "/Users/dillonragar/data/cpra/data_staging/d3d_veloc_000_zarr/AMP_D3D_WY22_000_FX_99_99_DLY_G900_AB_O_VELOCITY_V1.zarr"
DEM_PATH = f"{DATA_DIR}/60m_dem_1280_3200_padded.tif"
DOMAIN_PATH = f"{DATA_DIR}/D3D_model_domain.tif"
MODEL_PATH = "/Users/dillonragar/data/cpra/ml_out/xgb_dissolved_oxygen.json"
OUTPUT_DIR = f"{DATA_DIR}/data_staging/do"
OUTPUT_NC_PATH = f"{OUTPUT_DIR}/do_daily_WY22_000.nc"
OUTPUT_COG_DIR = f"{OUTPUT_DIR}/do_daily_WY22_000_cogs"


def drop_leap_days(ds: xr.Dataset) -> xr.Dataset:
    """Remove Feb 29 timesteps from a dataset."""
    time = ds["time"].dt
    mask = ~((time.month == 2) & (time.day == 29))
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
    water_year: int = None,
    overwrite: bool = False,
):
    """Save each variable and timestep as a Cloud Optimized GeoTIFF.

    Mirrors VegProcessor.utils.save_variables_as_cogs.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Reprojecting to web mercator...")
    ds = ds.rio.reproject("EPSG:3857")

    num_timesteps = len(ds.time.values)
    is_daily_data = num_timesteps == 365

    if is_daily_data and water_year is None:
        wy_match = re.search(r"WY(\d+)", output_dir)
        if wy_match:
            wy = int(wy_match.group(1))
            water_year = 2000 + wy if wy < 50 else 1900 + wy
        else:
            water_year = 2020

    for var_name in ds.data_vars:
        var = ds[var_name]
        var_dir = os.path.join(output_dir, var_name)
        os.makedirs(var_dir, exist_ok=True)
        print(f"\nProcessing variable: {var_name}")

        for t_idx in range(len(ds.time.values)):
            file_label = str(t_idx)

            da_slice = var.isel(time=t_idx).squeeze(drop=True)
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
    print("Done exporting COGs.")


def save_input_cogs(
    temp_ds: xr.Dataset,
    depth_da: xr.DataArray,
    vel_ds: xr.Dataset,
    output_dir: str,
    water_year: int = None,
    overwrite: bool = False,
):
    """Save DO model input variables as daily COGs for QAQC.

    Uses the same export pattern as
    ``VegProcessor.utils.save_variables_as_cogs``: reproject to Web
    Mercator, label files by water-year DOY, and write with
    odc.geo ``write_cog``.

    Parameters
    ----------
    temp_ds : xr.Dataset
        Temperature dataset with a ``temperature`` variable.
    depth_da : xr.DataArray
        Water depth array with dims (time, y, x).
    vel_ds : xr.Dataset
        Velocity dataset with a ``velocity`` variable.
    output_dir : str
        Root directory for the COG output.
    water_year : int, optional
        Water year used for DOY file labels. If None, extracted from
        ``output_dir`` path.
    overwrite : bool
        If True, overwrite existing COGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    inputs_ds = xr.Dataset(
        {
            "temperature": temp_ds["temperature"],
            "water_depth": depth_da,
            "velocity": vel_ds["velocity"],
        }
    )
    inputs_ds = inputs_ds.rio.write_crs("EPSG:6344")

    print("Reprojecting inputs to web mercator...")
    inputs_ds = inputs_ds.rio.reproject("EPSG:3857")

    # resolve water year
    if water_year is None:
        wy_match = re.search(r"WY(\d+)", output_dir)
        if wy_match:
            wy = int(wy_match.group(1))
            water_year = 2000 + wy if wy < 50 else 1900 + wy
        else:
            water_year = 2020

    for var_name in inputs_ds.data_vars:
        var = inputs_ds[var_name]
        var_dir = os.path.join(output_dir, var_name)
        os.makedirs(var_dir, exist_ok=True)
        print(f"\nProcessing variable: {var_name}")

        for t_idx in range(len(inputs_ds.time.values)):
            doy = t_idx + 1
            file_label = f"WY{water_year}_DOY_{doy:03d}"

            da_slice = var.isel(time=t_idx).squeeze(drop=True)
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
    print("Done exporting input COGs.")


def predict_do():
    """Run daily dissolved oxygen prediction and save to NetCDF."""
    import time as _time

    # Load DEM
    print("Loading DEM...")
    dem = xr.open_dataarray(DEM_PATH)
    dem = dem.squeeze(drop="band")
    dem = dem.rio.write_crs("EPSG:6344")

    # Load hydro domain mask (same as VegTransition / HSI)
    print("Loading hydro domain mask...")
    domain = xr.open_dataarray(DOMAIN_PATH)
    domain = domain.squeeze(drop="band")
    domain = domain.rio.write_crs("EPSG:6344")
    domain = domain.rio.reproject_match(dem)
    domain_mask = ~np.isnan(domain.values)  # True = valid domain pixel

    # Load XGBoost model
    print(f"Loading DO model from {MODEL_PATH}...")
    xgb = XGBRegressor()
    xgb.load_model(MODEL_PATH)

    # Load temperature eagerly (avoid dask scheduler overhead)
    print(f"Loading temperature from {TEMPERATURE_PATH}...")
    temp_ds = load_zarr_with_crs(TEMPERATURE_PATH)
    if "wtemp" in temp_ds.data_vars:
        temp_ds = temp_ds.rename({"wtemp": "temperature"})
    temp_ds = reproject_match(temp_ds, dem)
    temp_ds = drop_leap_days(temp_ds)
    temp = temp_ds["temperature"].load()  # (time, y, x) — into memory

    # Load depth (stage -> depth via DEM subtraction), eagerly
    print(f"Loading stage/depth from {DEPTH_PATH}...")
    stage_ds = load_zarr_with_crs(DEPTH_PATH)
    stage_ds = reproject_match(stage_ds, dem)
    stage_ds = drop_leap_days(stage_ds)
    depth = compute_depth(stage_ds, dem).load()  # (time, y, x) — into memory

    # Load velocity (daily timesteps)
    if VELOCITY_PATH is not None:
        print(f"Loading velocity from {VELOCITY_PATH}...")
        vel_ds = load_zarr_with_crs(VELOCITY_PATH)
        vel_ds = reproject_match(vel_ds, dem)
        vel_ds = drop_leap_days(vel_ds)
        if vel_ds["velocity"].sizes.get("time", 1) <= 1:
            raise ValueError(
                f"Velocity data must have daily timesteps, "
                f"but only has {vel_ds['velocity'].sizes.get('time', 0)} timestep(s). "
                f"Source: {VELOCITY_PATH}"
            )
        vel_vals = vel_ds["velocity"].load().values  # (time, y, x)
    else:
        vel_vals = None

    # Align all inputs to common timestamps
    common_times = np.intersect1d(temp.time.values, depth.time.values)
    if vel_vals is not None:
        common_times = np.intersect1d(
            common_times, vel_ds["velocity"].time.values
        )
    print(f"Common timesteps across all inputs: {len(common_times)}")
    temp_ds = temp_ds.sel(time=common_times)
    temp = temp_ds["temperature"].load()
    depth = depth.sel(time=common_times)
    if vel_vals is not None:
        vel_ds = vel_ds.sel(time=common_times)
        vel_vals = vel_ds["velocity"].load().values

    # Pre-extract numpy arrays and time metadata
    temp_vals = temp.values  # (time, y, x)
    depth_vals = depth.values  # (time, y, x)
    ny, nx = temp_vals.shape[1], temp_vals.shape[2]
    n_pixels = ny * nx

    print("Running DO prediction...")
    times = temp.time.values
    n_days = len(times)
    do_arr = np.empty((n_days, ny, nx), dtype=np.float32)
    t_start = _time.time()

    for i in range(n_days):
        t = times[i]
        ts = pd.Timestamp(t)
        month = ts.month
        doy = ts.dayofyear

        vel_flat = (
            vel_vals[i].ravel()
            if vel_vals is not None
            else np.zeros(n_pixels, dtype=np.float32)
        )

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
            elapsed = _time.time() - t_start
            per_day = elapsed / (i + 1)
            remaining = per_day * (n_days - i - 1)
            date_str = str(t)[:10]
            print(
                f"  [{i + 1}/{n_days}] {date_str} — "
                f"{elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining"
            )

    # Build xarray DataArray from numpy result
    do_da = xr.DataArray(
        do_arr,
        dims=("time", "y", "x"),
        coords={"time": times, "y": temp.y, "x": temp.x},
        attrs={
            "units": "mg/L",
            "long_name": "dissolved oxygen",
            "description": "Daily dissolved oxygen predicted by XGBoost model",
        },
    )
    do_ds = xr.Dataset({"dissolved_oxygen": do_da})
    do_ds = do_ds.rio.write_crs("EPSG:6344")

    # Write NetCDF
    print(f"Writing NetCDF to {OUTPUT_NC_PATH}...")
    do_ds.to_netcdf(
        OUTPUT_NC_PATH,
        encoding={
            "dissolved_oxygen": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
            },
            "time": {
                "units": "days since 1850-01-01T00:00:00",
                "calendar": "gregorian",
                "dtype": "float64",
            },
        },
    )

    # Write daily COGs (DO output)
    print(f"Writing daily COGs to {OUTPUT_COG_DIR}...")
    save_daily_cogs(do_ds, OUTPUT_COG_DIR, water_year=2020, overwrite=True)

    # Write daily COGs for input variables (QAQC)
    input_cog_dir = os.path.join(OUTPUT_DIR, "do_inputs_cogs")
    print(f"Writing input COGs to {input_cog_dir}...")
    save_input_cogs(
        temp_ds=temp_ds,
        depth_da=depth,
        vel_ds=vel_ds,
        output_dir=input_cog_dir,
        overwrite=True,
    )
    print("Done.")


if __name__ == "__main__":
    predict_do()
