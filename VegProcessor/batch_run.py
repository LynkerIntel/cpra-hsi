import os

from veg_transition import VegTransition
from hsi import HSI

veg_config_files = [
    # D3D
    # "/Users/dillonragar/data/cpra/configs/veg_d3d_config_1-08ft_slr_dry.yaml",
    # "/Users/dillonragar/data/cpra/configs/veg_d3d_config_1-08ft_slr_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/veg_d3d_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/veg_d3d_config_base_wet.yaml",
    # HEC
    "/Users/dillonragar/data/cpra/configs/veg_hec_config_1-08ft_slr_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/veg_hec_config_1-08ft_slr_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/veg_hec_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/veg_hec_config_base_wet.yaml",
]


# list of config files for each HSI run
hsi_config_files = [
    # D3D
    # "/Users/dillonragar/data/cpra/configs/hsi_d3d_config_1-08ft_dry.yaml",
    # "/Users/dillonragar/data/cpra/configs/hsi_d3d_config_1-08ft_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/hsi_d3d_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/hsi_d3d_config_base_wet.yaml",
    # HEC
    "/Users/dillonragar/data/cpra/configs/hsi_hec_config_1-08ft_slr_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/hsi_hec_config_1-08ft_slr_wet.yaml",
    "/Users/dillonragar/data/cpra/configs/hsi_hec_config_base_dry.yaml",
    "/Users/dillonragar/data/cpra/configs/hsi_hec_config_base_wet.yaml",
]


def main():
    run_veg = (
        input("Do you want to run Veg models? (y/n): ").lower().strip() == "y"
    )
    run_hsi = (
        input("Do you want to run HSI models? (y/n): ").lower().strip() == "y"
    )

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


if __name__ == "__main__":
    main()
