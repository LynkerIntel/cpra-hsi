import os
import yaml
import xarray as xr
import numpy as np

from veg_transition import VegTransition
from hsi import HSI
import utils

veg_config_files = [
    # D3D
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_d3d_config_1-08ft_slr_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_d3d_config_1-08ft_slr_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_d3d_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_d3d_config_base_wet.yaml",
    # HEC
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_hec_config_1-08ft_slr_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_hec_config_1-08ft_slr_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_hec_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_hec_config_base_wet.yaml",
    # MIKE
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_mik_config_1-08ft_slr_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_mik_config_1-08ft_slr_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_mik_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/veg_mik_config_base_wet.yaml",
]


# list of config files for each HSI run
hsi_config_files = [
    # D3D
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_d3d_config_1-08ft_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_d3d_config_1-08ft_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_d3d_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_d3d_config_base_wet.yaml",
    # HEC
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_hec_config_1-08ft_slr_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_hec_config_1-08ft_slr_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_hec_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_hec_config_base_wet.yaml",
    # MIK
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_mik_config_1-08ft_slr_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_mik_config_1-08ft_slr_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_mik_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/padded_dem/hsi_mik_config_base_wet.yaml",
]


def get_last_log_entries(
    output_dir: str, filename: str, n_lines: int = 2
) -> list:
    """Read the last n entries from the simulation log file."""
    log_path = os.path.join(
        output_dir, "run-metadata", f"{filename}_simulation.log"
    )
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-n_lines:] if line.strip()]
    except Exception:
        return []


def validate_veg_output(config_path: str) -> dict:
    """Check if VegTransition output completed successfully.

    Returns dict with 'success' bool, 'message' str, and 'log_entries' list.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Get expected parameters
        water_year_start = config["simulation"]["water_year_start"]
        water_year_end = config["simulation"]["water_year_end"]
        output_base = config["output"]["output_base"]
        sim_length = water_year_end - water_year_start

        # Build file params to match VegTransition.__init__
        metadata = config.get("metadata", {})
        file_params = {
            "model": metadata.get("model", "VEG"),
            "hydro_source_model": metadata.get("hydro_source_model"),
            "water_year": "WY99",
            "sea_level_condition": metadata.get("sea_level_condition"),
            "flow_scenario": metadata.get("flow_scenario"),
            "output_group": metadata.get("output_group"),
            "wpu": "AB",
            "io_type": "O",
            "time_freq": "ANN",
            "year_range": f"00_{str(sim_length + 1).zfill(2)}",
            "output_version": metadata.get("output_version"),
        }

        filename = utils.generate_filename(
            params=file_params,
            hydro_source_model=metadata.get("hydro_source_model"),
        )

        # Look for output directory and NetCDF file
        output_dir = os.path.join(output_base, str(filename))
        nc_path = os.path.join(output_dir, f"{filename}.nc")

        if not os.path.exists(nc_path):
            log_entries = get_last_log_entries(output_dir, str(filename))
            return {
                "success": False,
                "message": f"NetCDF not found",
                "log_entries": log_entries,
            }

        # Check time dimension
        with xr.open_dataset(nc_path) as ds:
            expected_timesteps = (
                water_year_end - water_year_start + 2
            )  # includes IC year
            actual_timesteps = ds.sizes["time"]

            if actual_timesteps != expected_timesteps:
                log_entries = get_last_log_entries(output_dir, str(filename))
                return {
                    "success": False,
                    "message": (
                        f"Time dim mismatch: expected {expected_timesteps}, got {actual_timesteps}"
                    ),
                    "log_entries": log_entries,
                }

            # Check last timestep has data (not all NaN)
            last_veg = ds["veg_type"].isel(time=-1).values
            if np.all(np.isnan(last_veg)):
                log_entries = get_last_log_entries(output_dir, str(filename))
                return {
                    "success": False,
                    "message": "Last timestep is all NaN",
                    "log_entries": log_entries,
                }

        return {"success": True, "message": "OK", "log_entries": []}

    except Exception as e:
        return {"success": False, "message": str(e), "log_entries": []}


def validate_hsi_output(config_path: str) -> dict:
    """Check if HSI output completed successfully.

    Returns dict with 'success' bool, 'message' str, and 'log_entries' list.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Get expected parameters
        water_year_start = config["simulation"]["water_year_start"]
        water_year_end = config["simulation"]["water_year_end"]
        output_base = config["output"]["output_base"]
        sim_length = water_year_end - water_year_start + 1  # HSI includes +1

        # Build file params to match HSI._load_config_attributes
        metadata = config.get("metadata", {})
        file_params = {
            "model": metadata.get("model", "HSI"),
            "hydro_source_model": metadata.get("hydro_source_model"),
            "water_year": "WY99",
            "sea_level_condition": metadata.get("sea_level_condition"),
            "flow_scenario": metadata.get("flow_scenario"),
            "output_group": metadata.get("output_group"),
            "wpu": "AB",
            "io_type": "O",
            "time_freq": "ANN",
            "year_range": f"01_{str(sim_length).zfill(2)}",  # HSI starts at 01
            "output_version": metadata.get("output_version"),
        }

        filename = utils.generate_filename(
            params=file_params,
            hydro_source_model=metadata.get("hydro_source_model"),
        )

        # Look for output directory and NetCDF files
        output_dir = os.path.join(output_base, str(filename))
        nc_480m_path = os.path.join(output_dir, f"{filename}.nc")
        nc_60m_path = os.path.join(output_dir, f"{filename}_60m.nc")

        # Check 480m file
        if not os.path.exists(nc_480m_path):
            log_entries = get_last_log_entries(output_dir, str(filename))
            return {
                "success": False,
                "message": "480m NetCDF not found",
                "log_entries": log_entries,
            }

        # Check 60m file
        if not os.path.exists(nc_60m_path):
            log_entries = get_last_log_entries(output_dir, str(filename))
            return {
                "success": False,
                "message": "60m NetCDF not found",
                "log_entries": log_entries,
            }

        # Check time dimension on 480m file
        with xr.open_dataset(nc_480m_path) as ds:
            expected_timesteps = water_year_end - water_year_start + 1
            actual_timesteps = ds.sizes["time"]

            if actual_timesteps != expected_timesteps:
                log_entries = get_last_log_entries(output_dir, str(filename))
                return {
                    "success": False,
                    "message": (
                        f"Time dim mismatch: expected {expected_timesteps}, got {actual_timesteps}"
                    ),
                    "log_entries": log_entries,
                }

            # Check that at least one data variable has data in last timestep
            data_vars = [v for v in ds.data_vars if v != "spatial_ref"]
            if data_vars:
                last_data = ds[data_vars[0]].isel(time=-1).values
                if np.all(np.isnan(last_data)):
                    log_entries = get_last_log_entries(
                        output_dir, str(filename)
                    )
                    return {
                        "success": False,
                        "message": "Last timestep is all NaN",
                        "log_entries": log_entries,
                    }

        return {"success": True, "message": "OK", "log_entries": []}

    except Exception as e:
        return {"success": False, "message": str(e), "log_entries": []}


def print_results_summary(veg_results: dict, hsi_results: dict):
    """Print a summary of successful and failed runs."""
    print("\n" + "=" * 60)
    print("BATCH RUN RESULTS SUMMARY")
    print("=" * 60)

    if veg_results:
        print("\n--- VegTransition Results ---")
        successful = [c for c, r in veg_results.items() if r["success"]]
        failed = [c for c, r in veg_results.items() if not r["success"]]

        print(f"\nSuccessful ({len(successful)}/{len(veg_results)}):")
        for config in successful:
            print(f"  ✓ {os.path.basename(config)}")

        if failed:
            print(f"\nFailed ({len(failed)}/{len(veg_results)}):")
            for config in failed:
                print(f"  ✗ {os.path.basename(config)}")
                print(f"    Reason: {veg_results[config]['message']}")
                log_entries = veg_results[config].get("log_entries", [])
                if log_entries:
                    print("    Last log entries:")
                    for entry in log_entries:
                        print(f"      {entry}")

    if hsi_results:
        print("\n--- HSI Results ---")
        successful = [c for c, r in hsi_results.items() if r["success"]]
        failed = [c for c, r in hsi_results.items() if not r["success"]]

        print(f"\nSuccessful ({len(successful)}/{len(hsi_results)}):")
        for config in successful:
            print(f"  ✓ {os.path.basename(config)}")

        if failed:
            print(f"\nFailed ({len(failed)}/{len(hsi_results)}):")
            for config in failed:
                print(f"  ✗ {os.path.basename(config)}")
                print(f"    Reason: {hsi_results[config]['message']}")
                log_entries = hsi_results[config].get("log_entries", [])
                if log_entries:
                    print("    Last log entries:")
                    for entry in log_entries:
                        print(f"      {entry}")

    print("\n" + "=" * 60)


def main():
    run_veg = (
        input("Do you want to run Veg models? (y/n): ").lower().strip() == "y"
    )
    run_hsi = (
        input("Do you want to run HSI models? (y/n): ").lower().strip() == "y"
    )
    validate_only = False
    if not run_veg and not run_hsi:
        validate_only = (
            input("Do you want to validate existing outputs? (y/n): ")
            .lower()
            .strip()
            == "y"
        )

    veg_results = {}
    hsi_results = {}

    if validate_only:
        print("\nValidating VegTransition outputs...")
        for config in veg_config_files:
            veg_results[config] = validate_veg_output(config)

        print("\nValidating HSI outputs...")
        for config in hsi_config_files:
            hsi_results[config] = validate_hsi_output(config)

        print_results_summary(veg_results, hsi_results)
        return

    if run_veg:
        # run each VegTransition scenario
        for config in veg_config_files:
            try:
                print(f"Running VegTransition model for config: {config}")
                veg = VegTransition(config_file=config)
                veg.run()
                veg.post_process()
                print(
                    f"Successfully completed VegTransition model for: {config}"
                )
            except Exception as e:
                print(
                    f"ERROR: VegTransition model failed for config: {config}"
                )
                print(f"Error message: {e}")
                print("Continuing to next config...")
                continue

        # Validate all veg outputs
        print("\nValidating VegTransition outputs...")
        for config in veg_config_files:
            veg_results[config] = validate_veg_output(config)

    if run_hsi:
        # run each HSI scenario
        for config in hsi_config_files:
            try:
                print(f"Running HSI model for config: {config}")
                hsi = HSI(config_file=config)
                hsi.run()
                hsi.post_process()
                print(f"Successfully completed HSI model for: {config}")
            except Exception as e:
                print(f"ERROR: HSI model failed for config: {config}")
                print(f"Error message: {e}")
                print("Continuing to next config...")
                continue

        # Validate all HSI outputs
        print("\nValidating HSI outputs...")
        for config in hsi_config_files:
            hsi_results[config] = validate_hsi_output(config)

    # Print summary
    if veg_results or hsi_results:
        print_results_summary(veg_results, hsi_results)


if __name__ == "__main__":
    main()
