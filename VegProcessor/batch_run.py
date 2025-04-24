import os

from veg_transition import VegTransition
from hsi import HSI

veg_config_files = [
    "./configs/veg_config_1-08ft_slr_dry.yaml",
    "./configs/veg_config_1-08ft_slr_moderate.yaml",
    "./configs/veg_config_1-08ft_slr_wet.yaml",
    #
    "./configs/veg_config_2ft_slr_dry.yaml",
    "./configs/veg_config_2ft_slr_moderate.yaml",
    "./configs/veg_config_2ft_slr_wet.yaml",
    #
    "./configs/veg_config_base_dry.yaml",
    "./configs/veg_config_base_moderate.yaml",
    "./configs/veg_config_base_wet.yaml",
]


# list of config files for each HSI run
hsi_config_files = [
    "./configs/hsi_config_1-08ft_slr_dry.yaml",
    "./configs/hsi_config_1-08ft_slr_moderate.yaml",
    "./configs/hsi_config_1-08ft_slr_wet.yaml",
    #
    "./configs/hsi_config_2ft_slr_dry.yaml",
    "./configs/hsi_config_2ft_slr_moderate.yaml",
    "./configs/hsi_config_2ft_slr_wet.yaml",
    #
    "./configs/hsi_config_base_dry.yaml",
    "./configs/hsi_config_base_moderate.yaml",
    "./configs/hsi_config_base_wet.yaml",
]


def main():
    # run each VegTransition scenario
    for config in veg_config_files:
        print(f"Running VegTransition model for config: {config}")
        Veg = VegTransition(config_file=config)
        Veg.run()
        Veg.post_process()

    # run each HSI scenario
    for config in hsi_config_files:
        print(f"Running HSI model for config: {config}")
        hsi = HSI(config_file=config)
        hsi.run()
        hsi.post_process()


if __name__ == "__main__":
    main()
