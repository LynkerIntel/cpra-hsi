import argparse
import glob
import os
import subprocess
import sys
import yaml
import xarray as xr
import numpy as np

from veg_transition import VegTransition
from hsi import HSI
import utils


# Variables written before the run loop (initial conditions / static) —
# their presence alone does NOT indicate the step loop ran. A run that
# aborts during the first step() still writes these.
VEG_STATIC_VARS = frozenset({"veg_type", "maturity", "spatial_ref", "crs"})
HSI_STATIC_VARS = frozenset({"spatial_ref", "crs"})


def _check_dynamic_output(ds: xr.Dataset, static_vars: frozenset) -> str | None:
    """Return an error message if the dataset has no populated dynamic vars, else None.

    A dataset is considered hollow if, after excluding known static/IC-only
    variables, no data variables remain, OR every remaining variable is
    entirely NaN. Either condition indicates the run loop never produced
    real output.
    """
    dynamic_vars = [
        v for v in ds.data_vars
        if v not in static_vars and "time" in ds[v].dims
    ]
    if not dynamic_vars:
        return (
            f"No dynamic output variables written "
            f"(found only {sorted(ds.data_vars)}); run likely aborted "
            f"before first step completed"
        )
    for v in dynamic_vars:
        if not np.all(np.isnan(ds[v].values)):
            return None
    return (
        f"All {len(dynamic_vars)} dynamic variables are entirely NaN; "
        f"run produced no populated output"
    )


def discover_configs(config_dir: str) -> tuple[list[str], list[str]]:
    """Find all veg and hsi yaml config files in the given directory tree.

    Searches recursively, so a parent directory holding one subdirectory per
    run group (e.g. `staging_pool/G400/veg_*.yaml`) can be passed directly.
    Sorting by path keeps configs grouped by subdirectory.
    """
    all_yamls = sorted(
        glob.glob(os.path.join(config_dir, "**", "*.yaml"), recursive=True)
    )
    veg_configs = [f for f in all_yamls if os.path.basename(f).startswith("veg_")]
    hsi_configs = [f for f in all_yamls if os.path.basename(f).startswith("hsi_")]
    return veg_configs, hsi_configs


def config_label(config_path: str, config_dir: str | None = None) -> str:
    """Return a display label for a config.

    Relative to `config_dir` when given, so configs with identical filenames in
    different run subdirectories stay distinguishable in logs and summaries.
    """
    if config_dir is None:
        return os.path.basename(config_path)
    try:
        return os.path.relpath(config_path, config_dir)
    except ValueError:  # different drives on Windows
        return config_path


def print_config_tree(config_files: list[str], config_dir: str) -> None:
    """Print a per-subdirectory veg/hsi count when configs are nested.

    No-op for a flat config dir, where the totals already say everything.
    """
    counts: dict[str, list[int]] = {}
    for config in config_files:
        subdir = os.path.relpath(os.path.dirname(config), config_dir)
        veg_hsi = counts.setdefault(subdir, [0, 0])
        if os.path.basename(config).startswith("veg_"):
            veg_hsi[0] += 1
        else:
            veg_hsi[1] += 1

    if set(counts) <= {"."}:
        return

    for subdir in sorted(counts):
        n_veg, n_hsi = counts[subdir]
        print(f"  {subdir}: {n_veg} veg, {n_hsi} hsi")


def get_current_branch() -> str:
    """Return the current git branch name, or 'HEAD' if detached."""
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def read_code_branch(config_path: str) -> str | None:
    """Return metadata.code_branch from a config, or None if unset/empty."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    metadata = config.get("metadata") or {}
    value = metadata.get("code_branch")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def check_code_branch(
    config_files: list[str], config_dir: str | None = None
) -> None:
    """Abort if any config pins code_branch to something other than current branch.

    Configs without code_branch are skipped. Detached HEAD with any pin set
    is treated as a mismatch so the user is forced to make the intent explicit.
    """
    pins = [(c, read_code_branch(c)) for c in config_files]
    pinned = [(c, b) for c, b in pins if b is not None]
    if not pinned:
        return

    current = get_current_branch()
    mismatches = [(c, b) for c, b in pinned if b != current]
    if not mismatches:
        return

    if current == "HEAD":
        current_desc = "detached HEAD"
    else:
        current_desc = f"branch '{current}'"
    print(
        f"ERROR: {len(mismatches)} config(s) pin code_branch to a branch other "
        f"than the current {current_desc}:"
    )
    for config, expected in mismatches:
        print(f"  {config_label(config, config_dir)}: expects '{expected}'")
    print("Checkout the expected branch or clear code_branch to proceed.")
    sys.exit(1)


def expected_output(config_path: str, model_type: str) -> tuple[str, str]:
    """Return (output_dir, filename) the given config is expected to write to.

    Mirrors the filename construction in `VegTransition.__init__` /
    `HSI._load_config_attributes` so batch validation can find the output
    without running the model. `model_type` is "veg" or "hsi".
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    water_year_start = config["simulation"]["water_year_start"]
    water_year_end = config["simulation"]["water_year_end"]
    output_base = config["output"]["output_base"]
    metadata = config.get("metadata", {})

    if model_type == "veg":
        sim_length = water_year_end - water_year_start
        default_model = "VEG"
        # veg output starts at the initial-condition year (00)
        year_range = f"00_{str(sim_length + 1).zfill(2)}"
    else:
        sim_length = water_year_end - water_year_start + 1  # HSI includes +1
        default_model = "HSI"
        year_range = f"01_{str(sim_length).zfill(2)}"  # HSI starts at 01

    file_params = {
        "model": metadata.get("model", default_model),
        "hydro_source_model": metadata.get("hydro_source_model"),
        "water_year": "WY99",
        "sea_level_condition": metadata.get("sea_level_condition"),
        "flow_scenario": metadata.get("flow_scenario"),
        "output_group": metadata.get("output_group"),
        "wpu": "AB",
        "io_type": "O",
        "time_freq": "ANN",
        "year_range": year_range,
        "output_version": metadata.get("output_version"),
    }

    filename = str(
        utils.generate_filename(
            params=file_params,
            hydro_source_model=metadata.get("hydro_source_model"),
        )
    )
    return os.path.join(output_base, filename), filename


def check_output_collisions(
    veg_config_files: list[str],
    hsi_config_files: list[str],
    config_dir: str | None = None,
) -> None:
    """Abort if two configs in the batch resolve to the same output path.

    Nested run directories reuse the same config filenames across
    subdirectories, so a copied subdir whose `output_group` was never bumped
    would have one run silently overwrite the other's results — and both
    would then validate as successful.
    """
    seen: dict[str, list[str]] = {}
    for configs, model_type in (
        (veg_config_files, "veg"),
        (hsi_config_files, "hsi"),
    ):
        for config in configs:
            try:
                output_dir, _ = expected_output(config, model_type)
            except Exception:
                # Unreadable configs are reported by validation, not here.
                continue
            seen.setdefault(output_dir, []).append(config)

    collisions = {d: c for d, c in seen.items() if len(c) > 1}
    if not collisions:
        return

    print(
        f"ERROR: {len(collisions)} output path(s) are claimed by more than one "
        f"config; running the batch would overwrite results:"
    )
    for output_dir, configs in sorted(collisions.items()):
        print(f"  {output_dir}")
        for config in configs:
            print(f"    <- {config_label(config, config_dir)}")
    print(
        "Give each run a distinct metadata.output_group (or output_base) to "
        "proceed."
    )
    sys.exit(1)


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

        water_year_start = config["simulation"]["water_year_start"]
        water_year_end = config["simulation"]["water_year_end"]

        output_dir, filename = expected_output(config_path, "veg")
        nc_path = os.path.join(output_dir, f"{filename}.nc")

        if not os.path.exists(nc_path):
            log_entries = get_last_log_entries(output_dir, filename)
            return {
                "success": False,
                "message": "NetCDF not found",
                "log_entries": log_entries,
            }

        # Check time dimension
        with xr.open_dataset(nc_path) as ds:
            expected_timesteps = (
                water_year_end - water_year_start + 2
            )  # includes IC year
            actual_timesteps = ds.sizes["time"]

            if actual_timesteps != expected_timesteps:
                log_entries = get_last_log_entries(output_dir, filename)
                return {
                    "success": False,
                    "message": (
                        f"Time dim mismatch: expected {expected_timesteps}, got {actual_timesteps}"
                    ),
                    "log_entries": log_entries,
                }

            # Check that dynamic (time-stepped) outputs were actually written.
            # A run that aborts before the first step still writes veg_type /
            # maturity from initial conditions, so those alone aren't evidence
            # the run loop completed.
            err = _check_dynamic_output(ds, VEG_STATIC_VARS)
            if err is not None:
                log_entries = get_last_log_entries(output_dir, filename)
                return {
                    "success": False,
                    "message": err,
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

        water_year_start = config["simulation"]["water_year_start"]
        water_year_end = config["simulation"]["water_year_end"]

        output_dir, filename = expected_output(config_path, "hsi")
        nc_480m_path = os.path.join(output_dir, f"{filename}.nc")
        nc_60m_path = os.path.join(output_dir, f"{filename}_60m.nc")

        # Check 480m file
        if not os.path.exists(nc_480m_path):
            log_entries = get_last_log_entries(output_dir, filename)
            return {
                "success": False,
                "message": "480m NetCDF not found",
                "log_entries": log_entries,
            }

        # Check 60m file
        if not os.path.exists(nc_60m_path):
            log_entries = get_last_log_entries(output_dir, filename)
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
                log_entries = get_last_log_entries(output_dir, filename)
                return {
                    "success": False,
                    "message": (
                        f"Time dim mismatch: expected {expected_timesteps}, got {actual_timesteps}"
                    ),
                    "log_entries": log_entries,
                }

            err = _check_dynamic_output(ds, HSI_STATIC_VARS)
            if err is not None:
                log_entries = get_last_log_entries(output_dir, filename)
                return {
                    "success": False,
                    "message": err,
                    "log_entries": log_entries,
                }

        return {"success": True, "message": "OK", "log_entries": []}

    except Exception as e:
        return {"success": False, "message": str(e), "log_entries": []}


def _print_model_results(
    header: str, results: dict, config_dir: str | None
) -> None:
    """Print the successful/failed breakdown for one model type."""
    print(f"\n--- {header} ---")
    successful = [c for c, r in results.items() if r["success"]]
    failed = [c for c, r in results.items() if not r["success"]]

    print(f"\nSuccessful ({len(successful)}/{len(results)}):")
    for config in successful:
        print(f"  ✓ {config_label(config, config_dir)}")

    if failed:
        print(f"\nFailed ({len(failed)}/{len(results)}):")
        for config in failed:
            print(f"  ✗ {config_label(config, config_dir)}")
            print(f"    Reason: {results[config]['message']}")
            log_entries = results[config].get("log_entries", [])
            if log_entries:
                print("    Last log entries:")
                for entry in log_entries:
                    print(f"      {entry}")


def print_results_summary(
    veg_results: dict, hsi_results: dict, config_dir: str | None = None
) -> int:
    """Print a summary of successful and failed runs.

    Returns the total number of failed configs across both model types.
    """
    print("\n" + "=" * 60)
    print("BATCH RUN RESULTS SUMMARY")
    print("=" * 60)

    if veg_results:
        _print_model_results("VegTransition Results", veg_results, config_dir)

    if hsi_results:
        _print_model_results("HSI Results", hsi_results, config_dir)

    total_failed = sum(
        1 for r in list(veg_results.values()) + list(hsi_results.values())
        if not r["success"]
    )
    print("\n" + "=" * 60)
    if total_failed:
        print(f"!! {total_failed} CONFIG(S) FAILED — see above !!")
        print("=" * 60)
    return total_failed


def main():
    parser = argparse.ArgumentParser(
        description="Batch run VegTransition and HSI models from a config directory."
    )
    parser.add_argument(
        "config_dir",
        help=(
            "Directory containing veg_*.yaml and hsi_*.yaml config files. "
            "Searched recursively, so a parent of per-run subdirectories works."
        ),
    )
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    if not os.path.isdir(config_dir):
        print(f"ERROR: {config_dir} is not a valid directory.")
        return

    veg_config_files, hsi_config_files = discover_configs(config_dir)
    print(
        f"Found {len(veg_config_files)} veg configs and "
        f"{len(hsi_config_files)} hsi configs in {config_dir}"
    )
    print_config_tree(veg_config_files + hsi_config_files, config_dir)

    if not veg_config_files and not hsi_config_files:
        print("Nothing to do: no veg_*.yaml or hsi_*.yaml files found.")
        return

    check_code_branch(veg_config_files + hsi_config_files, config_dir)
    check_output_collisions(veg_config_files, hsi_config_files, config_dir)

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

        total_failed = print_results_summary(
            veg_results, hsi_results, config_dir
        )
        sys.exit(1 if total_failed else 0)

    if run_veg:
        veg_run_failures = {}
        # run each VegTransition scenario
        for config in veg_config_files:
            try:
                print(
                    "Running VegTransition model for config: "
                    f"{config_label(config, config_dir)}"
                )
                # `with` detaches the run's log handlers on the way out, so
                # the next config does not also write into this run's log
                with VegTransition(config_file=config) as veg:
                    veg.run()
                    veg.post_process()
                print(
                    "Successfully completed VegTransition model for: "
                    f"{config_label(config, config_dir)}"
                )
            except Exception as e:
                print(
                    "ERROR: VegTransition model failed for config: "
                    f"{config_label(config, config_dir)}"
                )
                print(f"Error message: {e}")
                print("Continuing to next config...")
                veg_run_failures[config] = str(e)
                continue

        # Validate all veg outputs
        print("\nValidating VegTransition outputs...")
        for config in veg_config_files:
            if config in veg_run_failures:
                veg_results[config] = {
                    "success": False,
                    "message": f"Runtime error: {veg_run_failures[config]}",
                    "log_entries": [],
                }
            else:
                veg_results[config] = validate_veg_output(config)

    if run_hsi:
        hsi_run_failures = {}
        # run each HSI scenario
        for config in hsi_config_files:
            try:
                print(
                    "Running HSI model for config: "
                    f"{config_label(config, config_dir)}"
                )
                with HSI(config_file=config) as hsi:
                    hsi.run()
                    hsi.post_process()
                print(
                    "Successfully completed HSI model for: "
                    f"{config_label(config, config_dir)}"
                )
            except Exception as e:
                print(
                    "ERROR: HSI model failed for config: "
                    f"{config_label(config, config_dir)}"
                )
                print(f"Error message: {e}")
                print("Continuing to next config...")
                hsi_run_failures[config] = str(e)
                continue

        # Validate all HSI outputs
        print("\nValidating HSI outputs...")
        for config in hsi_config_files:
            if config in hsi_run_failures:
                hsi_results[config] = {
                    "success": False,
                    "message": f"Runtime error: {hsi_run_failures[config]}",
                    "log_entries": [],
                }
            else:
                hsi_results[config] = validate_hsi_output(config)

    # Print summary
    if veg_results or hsi_results:
        total_failed = print_results_summary(
            veg_results, hsi_results, config_dir
        )
        sys.exit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
