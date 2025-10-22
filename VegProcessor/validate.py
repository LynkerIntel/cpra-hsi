"""Configuration validation module for VegTransition and HSI models.

This module provides validation functions to check configuration files for
common errors and inconsistencies before model runs begin.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List


def validate_config(config: Dict, config_file: str = None) -> None:
    """Validate a configuration dictionary for common errors.

    Raises ConfigValidationError if validation fails.

    Parameters
    ----------
    config : Dict
        Loaded configuration dictionary
    config_file : str, optional
        Path to config file (for error messages)
    """
    errors = []

    # Run validation checks
    errors.extend(validate_hydro_source_consistency(config, config_file))

    if errors:
        error_msg = "\n".join(errors)
        raise ConfigValidationError(
            f"Configuration validation failed:\n{error_msg}"
        )


def validate_hydro_source_consistency(
    config: Dict, config_file: str = None
) -> List[str]:
    """Validate that hydro_source_model matches the file paths.

    Checks that:
    1. If hydro_source_model is HEC, veg_type_path contains "VEGH"
    2. If hydro_source_model is D3D, veg_type_path contains "VEGD"
    3. If hydro_source_model is MIK, veg_type_path contains "VEGM"
    4. netcdf_hydro_path matches the hydro_source_model

    Parameters
    ----------
    config : Dict
        Loaded configuration dictionary
    config_file : str, optional
        Path to config file (for error messages)

    Returns
    -------
    List[str]
        List of error messages (empty if valid)
    """
    errors = []
    config_name = config_file if config_file else "config"

    # Get hydro source model
    hydro_source = (
        config.get("metadata", {}).get("hydro_source_model", "").upper()
    )

    if not hydro_source:
        errors.append(
            f"[{config_name}] Missing 'metadata.hydro_source_model' in config"
        )
        return errors

    # Valid hydro source models
    valid_sources = ["HEC", "D3D", "MIK"]
    if hydro_source not in valid_sources:
        errors.append(
            f"[{config_name}] Invalid hydro_source_model: '{hydro_source}'. "
            f"Must be one of: {valid_sources}"
        )
        return errors

    # Map hydro source to expected veg prefix
    veg_prefix_map = {
        "HEC": "VEGH",
        "D3D": "VEGD",
        "MIK": "VEGM",
    }
    expected_veg_prefix = veg_prefix_map.get(hydro_source)

    # Check veg_type_path
    veg_type_path = config.get("raster_data", {}).get("veg_type_path", "")
    if veg_type_path:
        veg_filename = Path(veg_type_path).name
        if expected_veg_prefix not in veg_filename:
            errors.append(
                f"[{config_name}] Hydro source model mismatch: "
                f"hydro_source_model is '{hydro_source}' but veg_type_path contains "
                f"'{veg_filename}' (expected '{expected_veg_prefix}' in filename). "
            )
    else:
        errors.append(
            f"[{config_name}] Missing 'raster_data.veg_type_path' in config"
        )

    # Check netcdf_hydro_path
    netcdf_hydro_path = config.get("raster_data", {}).get(
        "netcdf_hydro_path", ""
    )
    if netcdf_hydro_path:
        hydro_dir_name = Path(netcdf_hydro_path).name
        if hydro_source not in hydro_dir_name:
            errors.append(
                f"[{config_name}] Hydro source model mismatch: "
                f"hydro_source_model is '{hydro_source}' but netcdf_hydro_path contains "
                f"'{hydro_dir_name}' (expected '{hydro_source}' in path)"
            )
    else:
        errors.append(
            f"[{config_name}] Missing 'raster_data.netcdf_hydro_path' in config"
        )

    return errors


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass
