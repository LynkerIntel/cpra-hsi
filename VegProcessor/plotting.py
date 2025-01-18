import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
import pandas as pd
import os
import xarray as xr
from typing import Optional


logger = logging.getLogger("VegTransition")


def np_arr(
    arr: np.ndarray,
    title: str,
    veg_type_desc: Optional[str] = "",
    out_path: Optional[str] = None,
    showplot: Optional[bool] = False,
    veg_palette: Optional[bool] = False,
):
    """
    Plot 2D numpy arrays and their histogram using unique values for vegetation color palette.

    Parameters
    ----------
    arr : np.ndarray
        2D numpy array to be plotted.
    title : str
        Title for the plot.
    veg_type_desc : str
        Description of the input vegetation type to be included in the plot title.
    out_path : str, optional
        Directory to save the plot. If None, the plot is not saved.
    showplot : bool, optional
        If True, display the plot after generating it. Default is False.
    veg_palette : bool, optional
        If True, use the `vegetation_colors` dictionary to color the array and histogram based on vegetation types.

    Returns
    -------
    None
        Saves figures based on input arrays, in the `out_path`
    """
    # Define vegetation types and their associated colors
    vegetation_colors = {
        15: (0, 102, 0),  # Dark Green
        16: (0, 204, 153),  # Teal
        17: (0, 255, 153),  # Light Green
        18: (102, 153, 0),  # Olive Green
        19: (128, 128, 0),  # Dark Yellow
        20: (51, 204, 51),  # Light Green
        21: (255, 255, 102),  # Light Yellow
        22: (237, 125, 49),  # Orange
        23: (255, 0, 0),  # Red
        26: (153, 204, 255),  # Light Blue
    }

    unique_values = np.unique(arr[~np.isnan(arr)]).astype(int)

    if veg_palette:
        # Extract the keys and colors from the vegetation colors
        veg_keys = sorted(unique_values)  # Get all unique values from the array
        veg_colors = [
            tuple(np.array(vegetation_colors.get(key, (128, 128, 128))) / 255.0)
            for key in veg_keys
        ]  # Normalize to 0-1 and use gray (128, 128, 128) for missing keys
    else:
        veg_keys = sorted(unique_values)
        veg_colors = plt.cm.viridis(np.linspace(0, 1, len(veg_keys)))

    # Create a Normalize object to map vegetation type values directly to 0-1
    norm = mcolors.Normalize(vmin=min(veg_keys), vmax=max(veg_keys))

    # Create a ListedColormap from the specified vegetation colors
    cmap = mcolors.ListedColormap(veg_colors)

    # Plot the vegetation array
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(arr, cmap=cmap, norm=norm)
    axes[0].set_title(f"{title}\n{veg_type_desc}", fontsize=10)

    # # Create the colorbar and set tick positions at the unique vegetation type values
    # cbar = fig.colorbar(im, ax=axes[0], orientation="vertical")
    # cbar.set_ticks(veg_keys)  # Tick at each vegetation type
    # cbar.set_ticklabels(
    #     [str(veg_type) for veg_type in veg_keys]
    # )  # Label tick at each vegetation type

    # Generate a histogram of vegetation types in the array
    unique_types, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
    sorted_indices = np.argsort(unique_types)
    unique_types = unique_types[sorted_indices]
    counts = counts[sorted_indices]

    if veg_palette:
        bar_colors = [
            tuple(np.array(vegetation_colors.get(veg_type, (128, 128, 128))) / 255.0)
            for veg_type in unique_types
        ]
    else:
        bar_colors = ["blue"] * len(unique_types)

    # Plot the histogram of vegetation types
    axes[1].bar(unique_types, counts, color=bar_colors, align="center", alpha=0.7)
    axes[1].set_title(f"Histogram of Array Values\n{veg_type_desc}", fontsize=10)
    axes[1].set_xlabel("Vegetation Type")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xticks(unique_types)
    axes[1].set_xticklabels(
        [str(int(veg_type)) for veg_type in unique_types], rotation=45
    )

    plt.tight_layout()

    if out_path:
        os.makedirs(out_path, exist_ok=True)
        sanitized_title = title.replace(" ", "_").replace("\n", "_")
        file_path = os.path.join(out_path, f"{sanitized_title}.png")
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {file_path}")

    if showplot:
        plt.show()

    plt.close()


def sum_changes(
    input_array: np.ndarray,
    output_array: np.ndarray,
    plot_title: Optional[str] = "Vegetation Type Changes",
    out_path: Optional[str] = None,
    show_plot: Optional[bool] = False,
):
    """
    Calculate and plot the changes in vegetation type counts between input and output arrays.

    Parameters
    ----------
    input_array : np.ndarray
        The initial vegetation type array.
    output_array : np.ndarray
        The resulting vegetation type array after transitions.
    plot_title : str, optional
        Title for the plot. Default is "Vegetation Type Changes".
    out_path : str, optional
        Directory to save the plot. If None, the plot is not saved.
    show_plot : bool, optional
        If True, display the plot after generating it. Default is False.

    Returns
    -------
    None
        Saves the plot if `out_path` is specified and optionally displays the plot.
    """
    if input_array.shape != output_array.shape:
        raise ValueError("Input and output arrays must have the same shape.")

    # Count occurrences of each vegetation type in both arrays
    unique_types_input, counts_input = np.unique(
        input_array[~np.isnan(input_array)], return_counts=True
    )
    unique_types_output, counts_output = np.unique(
        output_array[~np.isnan(output_array)], return_counts=True
    )

    input_counts = dict(zip(unique_types_input, counts_input))
    output_counts = dict(zip(unique_types_output, counts_output))
    all_types = sorted(set(input_counts.keys()).union(output_counts.keys()))

    # Calculate differences
    differences = [
        output_counts.get(veg_type, 0) - input_counts.get(veg_type, 0)
        for veg_type in all_types
    ]

    # Plot the changes
    plt.figure(figsize=(10, 6))
    plt.bar(all_types, differences, tick_label=[int(veg) for veg in all_types])
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Vegetation Type")
    plt.ylabel("Change in Count")
    plt.title(plot_title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if out_path:
        os.makedirs(out_path, exist_ok=True)  # Ensure the directory exists
        sanitized_title = plot_title.replace(" ", "_").replace("\n", "_")
        file_path = os.path.join(out_path, f"{sanitized_title}.png")
        plt.savefig(file_path, dpi=300)
        plt.close(plt.gcf())
        logger.info(f"Saved plot to {file_path}")

    if show_plot:
        plt.show()

    plt.close()


def water_depth(
    ds: xr.Dataset,
    out_path: Optional[str] = None,
    showplot: Optional[bool] = False,
):
    """
    Create figure of water depth for a 12-month period with square pixels.
    """
    os.makedirs(out_path + "/water_depth/", exist_ok=True)

    time_steps = ds.time.values
    for i, t in enumerate(time_steps):
        date_str = pd.to_datetime(t).strftime("%Y-%m-%d")

        # Select data slice
        data_slice = ds["WSE_MEAN"].sel(time=t)

        # Create plot with Xarray managing the figure
        fig, ax = plt.subplots(figsize=(10, 10))  # Ensure square figure size
        data_slice.plot(
            ax=ax,
            robust=True,
            cbar_kwargs={"label": "water depth (m)"},  # Add color bar label here
        )

        # Add title and labels
        ax.set_title(
            f"Water Depth at Time {date_str}\n"
            "Color bar uses 2nd and 98th percentiles of data for color range.",
            fontsize=12,
        )
        ax.set_aspect("equal", "box")  # Forces square pixels regardless of figure size

        # Save or display the plot
        if out_path:
            file_path = os.path.join(out_path, f"water_depth/{date_str}.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {file_path}")
            plt.close(fig)
        else:
            plt.show()
