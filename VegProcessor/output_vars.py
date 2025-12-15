import numpy as np


def _safe_get_attr(obj, attr):
    """Safely get attribute from object, returning None if object is None."""
    return getattr(obj, attr) if obj is not None else None


def get_hsi_variables(hsi):
    """
    returns a dictionary where each key is a string representing a variable name (str),
    and each value is a list with the following structure:

    - data (numpy.ndarray or a similar object, typically retrieved as an attribute
        from the hsi model instance)
    - dtype (numpy data type, e.g., np.float32) used when writing the variable to a
        NetCDF file
    - metadata (dict[str, str]) containing NetCDF attribute metadata, including keys such as:
        - "grid_mapping": typically set to "spatial_ref"
        - "units": string representing the measurement units (e.g., "meters", "%", or an
            empty string if unitless)
        - "long_name": human-readable name of the variable
        - "description": optional extended description of the variable

    Only returns variables for species listed in hsi.hsi_run_species config.
    Variables for species not run will have None data and will be filtered by
    the NetCDF append function.
    """
    all_variables = {
        "alligator_hsi": [
            _safe_get_attr(hsi.alligator, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "alligator_si_1": [
            _safe_get_attr(hsi.alligator, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "alligator_si_2": [
            _safe_get_attr(hsi.alligator, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "alligator_si_3": [
            _safe_get_attr(hsi.alligator, "si_3"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "alligator_si_4": [
            _safe_get_attr(hsi.alligator, "si_4"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "alligator_si_5": [
            _safe_get_attr(hsi.alligator, "si_5"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bald_eagle_hsi": [
            _safe_get_attr(hsi.baldeagle, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bald_eagle_si_1": [
            _safe_get_attr(hsi.baldeagle, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bald_eagle_si_2": [
            _safe_get_attr(hsi.baldeagle, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bald_eagle_si_3": [
            _safe_get_attr(hsi.baldeagle, "si_3"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bald_eagle_si_4": [
            _safe_get_attr(hsi.baldeagle, "si_4"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bald_eagle_si_5": [
            _safe_get_attr(hsi.baldeagle, "si_5"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bald_eagle_si_6": [
            _safe_get_attr(hsi.baldeagle, "si_6"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "crawfish_hsi": [
            _safe_get_attr(hsi.crawfish, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "crawfish_si_1": [
            _safe_get_attr(hsi.crawfish, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "crawfish_si_2": [
            _safe_get_attr(hsi.crawfish, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "crawfish_si_3": [
            _safe_get_attr(hsi.crawfish, "si_3"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "crawfish_si_4": [
            _safe_get_attr(hsi.crawfish, "si_4"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "gizzard_shad_hsi": [
            _safe_get_attr(hsi.gizzardshad, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "gizzard_shad_si_1": [
            _safe_get_attr(hsi.gizzardshad, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "gizzard_shad_si_2": [
            _safe_get_attr(hsi.gizzardshad, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "gizzard_shad_si_3": [
            _safe_get_attr(hsi.gizzardshad, "si_3"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "gizzard_shad_si_4": [
            _safe_get_attr(hsi.gizzardshad, "si_4"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "gizzard_shad_si_5": [
            _safe_get_attr(hsi.gizzardshad, "si_5"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "gizzard_shad_si_6": [
            _safe_get_attr(hsi.gizzardshad, "si_6"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "gizzard_shad_si_7": [
            _safe_get_attr(hsi.gizzardshad, "si_7"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bass_hsi": [
            _safe_get_attr(hsi.bass, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bass_si_1": [
            _safe_get_attr(hsi.bass, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bass_si_2": [
            _safe_get_attr(hsi.bass, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "catfish_hsi": [
            _safe_get_attr(hsi.catfish, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "riverine catfish HSI",
                "description": "",
            },
        ],
        "catfish_si_1": [
            _safe_get_attr(hsi.catfish, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "percent pools during average summer flow",
            },
        ],
        "catfish_si_2": [
            _safe_get_attr(hsi.catfish, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "percent cover in summer pools backwater",
            },
        ],
        "catfish_si_4": [
            _safe_get_attr(hsi.catfish, "si_4"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "food production potential substrate",
            },
        ],
        "catfish_si_5": [
            _safe_get_attr(hsi.catfish, "si_5"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": (
                    "average midsummer temperature in pools backwater (adult)"
                ),
            },
        ],
        "catfish_si_6": [
            _safe_get_attr(hsi.catfish, "si_6"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "growing season length frost free days",
            },
        ],
        "catfish_si_7": [
            _safe_get_attr(hsi.catfish, "si_7"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "maximum monthly average summer turbidity",
            },
        ],
        "catfish_si_8": [
            _safe_get_attr(hsi.catfish, "si_8"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": (
                    "average minimum dissolved oxygen in midsummer pools backwater"
                ),
            },
        ],
        "catfish_si_9": [
            _safe_get_attr(hsi.catfish, "si_9"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "maximum salinity during summer (adult)",
            },
        ],
        "catfish_si_10": [
            _safe_get_attr(hsi.catfish, "si_10"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": (
                    "average temperature in spawning embryo pools backwater (embryo)"
                ),
            },
        ],
        "catfish_si_11": [
            _safe_get_attr(hsi.catfish, "si_11"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "maximum salinity spawning embryo (embryo)",
            },
        ],
        "catfish_si_12": [
            _safe_get_attr(hsi.catfish, "si_12"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": (
                    "average midsummer temperature in pools backwater (fry)"
                ),
            },
        ],
        "catfish_si_13": [
            _safe_get_attr(hsi.catfish, "si_13"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "maximum summer salinity fry juvenile",
            },
        ],
        "catfish_si_14": [
            _safe_get_attr(hsi.catfish, "si_14"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": (
                    "average midsummer temperature in pools backwater (juvenile)"
                ),
            },
        ],
        "catfish_si_18": [
            _safe_get_attr(hsi.catfish, "si_18"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": (
                    "average current velocity in cover areas during average summer flow"
                ),
            },
        ],
        "catfish_fc": [
            _safe_get_attr(hsi.catfish, "fc"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "riverine catfish food component",
            },
        ],
        "catfish_cc": [
            _safe_get_attr(hsi.catfish, "cc"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "riverine catfish cover component",
            },
        ],
        "catfish_wq": [
            _safe_get_attr(hsi.catfish, "wq"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "riverine catfish water quality component",
            },
        ],
        "catfish_rc": [
            _safe_get_attr(hsi.catfish, "rc"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "riverine catfish reproduction component",
            },
        ],
        "blackbear_hsi": [
            _safe_get_attr(hsi.blackbear, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackbear_si_1": [
            _safe_get_attr(hsi.blackbear, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackbear_si_2": [
            _safe_get_attr(hsi.blackbear, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackbear_si_3": [
            _safe_get_attr(hsi.blackbear, "si_3"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackbear_si_4": [
            _safe_get_attr(hsi.blackbear, "si_4"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackbear_si_5": [
            _safe_get_attr(hsi.blackbear, "si_5"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackbear_si_6": [
            _safe_get_attr(hsi.blackbear, "si_6"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackbear_si_7": [
            _safe_get_attr(hsi.blackbear, "si_7"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackbear_si_8": [
            _safe_get_attr(hsi.blackbear, "si_8"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bluecrab_si_1": [
            _safe_get_attr(hsi.bluecrab, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bluecrab_si_2": [
            _safe_get_attr(hsi.bluecrab, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "bluecrab_hsi": [
            _safe_get_attr(hsi.bluecrab, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blh_wva_hsi": [
            _safe_get_attr(hsi.blhwva, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "bottomland_hardwood_wetland_value_assessment",
                "description": "",
            },
        ],
        "blh_wva_si_1": [
            _safe_get_attr(hsi.blhwva, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "bottomland hardwood wetland value assessment",
            },
        ],
        "blh_wva_si_2": [
            _safe_get_attr(hsi.blhwva, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "bottomland hardwood wetland value assessment",
            },
        ],
        "blh_wva_si_3": [
            _safe_get_attr(hsi.blhwva, "si_3"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "bottomland hardwood wetland value assessment",
            },
        ],
        "blh_wva_si_4": [
            _safe_get_attr(hsi.blhwva, "si_4"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "bottomland hardwood wetland value assessment",
            },
        ],
        "blh_wva_si_5": [
            _safe_get_attr(hsi.blhwva, "si_5"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "bottomland hardwood wetland value assessment",
            },
        ],
        "swamp_wva_hsi": [
            _safe_get_attr(hsi.swampwva, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "swamp wetland value assessment",
            },
        ],
        "swamp_wva_si_1": [
            _safe_get_attr(hsi.swampwva, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "swamp wetland value assessment",
            },
        ],
        "swamp_wva_si_2": [
            _safe_get_attr(hsi.swampwva, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "swamp wetland value assessment",
            },
        ],
        "swamp_wva_si_3": [
            _safe_get_attr(hsi.swampwva, "si_3"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "swamp wetland value assessment",
            },
        ],
        "swamp_wva_si_4": [
            _safe_get_attr(hsi.swampwva, "si_4"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "swamp wetland value assessment",
            },
        ],
        "swamp_wva_si_5": [
            _safe_get_attr(hsi.swampwva, "si_5"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "swamp wetland value assessment",
            },
        ],
        "swamp_wva_si_6": [
            _safe_get_attr(hsi.swampwva, "si_6"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "swamp wetland value assessment",
            },
        ],
        "swamp_wva_si_7": [
            _safe_get_attr(hsi.swampwva, "si_7"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "description": "swamp wetland value assessment",
            },
        ],
        "pct_open_water": [
            hsi.pct_open_water,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "salinity_annual_mean": [
            hsi.salinity_annual_mean,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "ppt",
                "long_name": "",
                "description": "",
            },
        ],
        "water_temperature_annual_mean": [
            hsi.water_temperature_annual_mean,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "Deg C",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_blh": [
            hsi.pct_blh,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_swamp_bottom_hardwood": [
            hsi.pct_swamp_bottom_hardwood,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_fresh_marsh": [
            hsi.pct_fresh_marsh,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_intermediate_marsh": [
            hsi.pct_intermediate_marsh,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_brackish_marsh": [
            hsi.pct_brackish_marsh,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_saline_marsh": [
            hsi.pct_saline_marsh,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_zone_v": [
            hsi.pct_zone_v,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_zone_iv": [
            hsi.pct_zone_iv,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_zone_iii": [
            hsi.pct_zone_iii,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_zone_ii": [
            hsi.pct_zone_ii,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_fresh_shrubs": [
            hsi.pct_fresh_shrubs,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_bare_ground": [
            hsi.pct_bare_ground,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_dev_upland": [
            hsi.pct_dev_upland,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_flotant_marsh": [
            hsi.pct_flotant_marsh,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_vegetated": [
            hsi.pct_vegetated,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_soft_mast": [
            hsi.pct_soft_mast,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_hard_mast": [
            hsi.pct_hard_mast,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_no_mast": [
            hsi.pct_no_mast,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_near_forest": [
            hsi.pct_near_forest,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_forested_half_mi": [
            hsi.pct_forested_half_mi,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_abandoned_ag_half_mi": [
            hsi.pct_abandoned_ag_half_mi,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_pasture_half_mi": [
            hsi.pct_pasture_half_mi,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_active_ag_water_half_mi": [
            hsi.pct_active_ag_water_half_mi,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_nonhabitat_half_mi": [
            hsi.pct_nonhabitat_half_mi,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "water_depth_annual_mean": [
            hsi.water_depth_annual_mean,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "",
                "description": "",
            },
        ],
        "water_depth_jan_aug_mean": [
            hsi.water_depth_jan_aug_mean,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "",
                "description": "",
            },
        ],
        "water_depth_oct_dec_mean": [
            hsi.water_depth_oct_dec_mean,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "",
                "description": "",
            },
        ],
        "water_depth_april_june_mean": [
            hsi.water_depth_april_june_mean,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "water depth April-June mean",
                "description": (
                    "Mean water depth for April-June period at 480m resolution"
                ),
            },
        ],
        "salinity_max_april_sept": [
            hsi.salinity_max_april_sept,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "ppt",
                "long_name": "salinity maximum April-September",
                "description": (
                    "Maximum salinity for April-September period at 480m resolution"
                ),
            },
        ],
        "salinity_max_july_sept": [
            hsi.salinity_max_july_sept,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "ppt",
                "long_name": "salinity maximum July-September",
                "description": (
                    "Maximum salinity for July-September period at 480m resolution"
                ),
            },
        ],
        "salinity_max_may_july": [
            hsi.salinity_max_may_july,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "ppt",
                "long_name": "salinity maximum May-July",
                "description": (
                    "Maximum salinity for May-July period at 480m resolution"
                ),
            },
        ],
        "salinity_mean_high_march_nov": [
            hsi.salinity_mean_high_march_nov,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "ppt",
                "long_name": "salinity mean high March-November",
                "description": (
                    "Mean of upper 67th percentile salinity values for March-November period at 480m resolution"
                ),
            },
        ],
        "pct_pools_july_sept_mean": [
            hsi.pct_pools_july_sept_mean,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "%",
                "long_name": "percent pools July-September mean",
                "description": (
                    "Percentage of pool pixels for July-September period"
                ),
            },
        ],
        "pct_pools_april_sept_mean": [
            hsi.pct_pools_april_sept_mean,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "%",
                "long_name": "percent pools April-September mean",
                "description": (
                    "Percentage of pool pixels for April-September period"
                ),
            },
        ],
        "pct_shrub_scrub": [
            hsi.pct_shrub_scrub,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "%",
                "long_name": "percent shrub/scrub cover",
                "description": "Percentage of shrub/scrub vegetation cover",
            },
        ],
        "edge": [
            hsi.edge,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "%",
                "long_name": "edge pixels",
                "description": "Percentage of water edge pixels (static)",
            },
        ],
        "pct_has_mast": [
            hsi.pct_has_mast,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "%",
                "long_name": "percent mast-producing vegetation",
                "description": (
                    "Percentage of vegetation with mast production (soft + hard mast combined)"
                ),
            },
        ],
        "pct_crops": [
            hsi.pct_crops,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "%",
                "long_name": "percent crops",
                "description": "Percentage of crop land cover (static)",
            },
        ],
        "pct_developed": [
            hsi.pct_developed,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "%",
                "long_name": "percent developed land",
                "description": "Percentage of developed land cover (static)",
            },
        ],
        "velocity": [
            hsi.velocity,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "m/s",
                "long_name": "water velocity",
                "description": "Mean water velocity at 480m resolution",
            },
        ],
        "human_influence_bool": [
            hsi.human_influence,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "forested_connectivity": [
            hsi.forested_connectivity_cat,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "forested_connectivity category",
                "description": "",
            },
        ],
        "story_class": [
            hsi.story_class,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "forested story classification (mode of 60m)",
                "description": "",
            },
        ],
        "pct_overstory": [
            hsi.pct_overstory,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_midstory": [
            hsi.pct_midstory,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_understory": [
            hsi.pct_understory,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "pct_shrub_scrub_midstory": [
            hsi.pct_shrub_scrub_midstory,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "summed pct cover of shrub/scrup & midstory",
                "description": "",
            },
        ],
        "blackcrappie_hsi": [
            _safe_get_attr(hsi.blackcrappie, "hsi"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "black crappie HSI",
                "description": "HSI",
            },
        ],
        "blackcrappie_si_1": [
            _safe_get_attr(hsi.blackcrappie, "si_1"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_2": [
            _safe_get_attr(hsi.blackcrappie, "si_2"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_3": [
            _safe_get_attr(hsi.blackcrappie, "si_3"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_4": [
            _safe_get_attr(hsi.blackcrappie, "si_4"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_5": [
            _safe_get_attr(hsi.blackcrappie, "si_5"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_7": [
            _safe_get_attr(hsi.blackcrappie, "si_7"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_8": [
            _safe_get_attr(hsi.blackcrappie, "si_8"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_9": [
            _safe_get_attr(hsi.blackcrappie, "si_9"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_10": [
            _safe_get_attr(hsi.blackcrappie, "si_10"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_11": [
            _safe_get_attr(hsi.blackcrappie, "si_11"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_12": [
            _safe_get_attr(hsi.blackcrappie, "si_12"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_si_13": [
            _safe_get_attr(hsi.blackcrappie, "si_13"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "",
            },
        ],
        "blackcrappie_fc": [
            _safe_get_attr(hsi.blackcrappie, "fc"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "black crappie food components",
            },
        ],
        "blackcrappie_wq_tcr": [
            _safe_get_attr(hsi.blackcrappie, "wq_tcr"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "black crappie water quality component",
            },
        ],
        "blackcrappie_wq_tcr_adj": [
            _safe_get_attr(hsi.blackcrappie, "wq_tcr_adj"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": (
                    "black crappie water quality component adjusted"
                ),
            },
        ],
        "blackcrappie_wq_init": [
            _safe_get_attr(hsi.blackcrappie, "wq_init"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "black crappie water quality initial",
            },
        ],
        "blackcrappie_wq": [
            _safe_get_attr(hsi.blackcrappie, "wq"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "black crappie water quality",
            },
        ],
        "blackcrappie_rc": [
            _safe_get_attr(hsi.blackcrappie, "rc"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "black crappie reproduction component",
            },
        ],
        "blackcrappie_ot": [
            _safe_get_attr(hsi.blackcrappie, "ot"),
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "",
                "description": "black crappie other component",
            },
        ],
    }

    return all_variables


def get_veg_variables(veg):
    """
    returns a dictionary where each key is a string representing a variable name (str),
    and each value is a list with the following structure:

    - data (numpy.ndarray or a similar object, typically retrieved as an attribute
        from the veg model instance)
    - dtype (numpy data type, e.g., np.float32) used when writing the variable to a
        NetCDF file
    - metadata (dict[str, str]) containing NetCDF attribute metadata, including keys such as:
        - "grid_mapping": typically set to "spatial_ref"
        - "units": string representing the measurement units (e.g., "meters", "%", or an
            empty string if unitless)
        - "long_name": human-readable name of the variable
        - "description": optional extended description of the variable
    """
    return {
        "veg_type": [
            veg.veg_type,
            np.float32,
            {
                "grid_mapping": "spatial_ref",  # Link CRS variable
                "units": "",
                "long_name": "veg type",
            },
        ],
        "maturity": [
            veg.maturity,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "years",
                "long_name": "forested vegetation age",
            },
        ],
        "salinity_annual_mean": [
            veg.salinity_annual_mean,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "ppt",
                "long_name": "water salinity",
            },
        ],
        # QC variables below
        "qc_annual_mean_salinity": [
            veg.qc_annual_mean_salinity,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "ppt",
                "long_name": "mean annual salinity",
            },
        ],
        "qc_annual_inundation_depth": [
            veg.qc_annual_inundation_depth,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "annual inundation depth",
            },
        ],
        "qc_annual_inundation_duration": [
            veg.qc_annual_inundation_duration,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "% time",
                "long_name": "annual inundation duration",
                "description": "Percentage of time flooded over the year",
            },
        ],
        "qc_growing_season_depth": [
            veg.qc_growing_season_depth,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "growing season depth",
                "description": (
                    "Average water-depth during the period from April 1 through September 30"
                ),
            },
        ],
        "qc_growing_season_inundation": [
            veg.qc_growing_season_inundation,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "% time",
                "long_name": "growing season inundation",
                "description": (
                    "Percentage of time flooded during the period from April 1 through September 30"
                ),
            },
        ],
        "qc_tree_establishment_bool": [
            veg.qc_tree_establishment_bool,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "",
                "long_name": "tree establishment (true or false)",
                "description": "Areas where establishment condition is met",
            },
        ],
        # Seasonal water depth QC variables
        "qc_march_water_depth": [
            veg.qc_march_water_depth,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "march water depth",
                "description": "Depth in m for the month of march.",
            },
        ],
        "qc_april_water_depth": [
            veg.qc_april_water_depth,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "april water depth",
                "description": "Depth in m for the month of april.",
            },
        ],
        "qc_may_water_depth": [
            veg.qc_may_water_depth,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "may water depth",
                "description": "Depth in m for the month of may.",
            },
        ],
        "qc_june_water_depth": [
            veg.qc_june_water_depth,
            np.float32,
            {
                "grid_mapping": "spatial_ref",
                "units": "meters",
                "long_name": "june water depth",
                "description": "Depth in m for the month of june.",
            },
        ],
    }
