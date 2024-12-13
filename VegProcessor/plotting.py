import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import logging
import os
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get all unique values in the array (this will be used as a fallback for veg_keys if veg_palette is False)
    unique_values = np.unique(arr[~np.isnan(arr)]).astype(int)

    if veg_palette:
        # Extract the vegetation type keys and colors
        veg_keys = list(vegetation_colors.keys())
        veg_colors = [
            np.array(color) / 255.0 for color in vegetation_colors.values()
        ]  # Normalize 0-1

        # Identify missing (undefined) vegetation types and assign them a gray color
        undefined_veg_types = [val for val in unique_values if val not in veg_keys]

        # Add gray color for undefined vegetation types
        for veg_type in undefined_veg_types:
            vegetation_colors[veg_type] = (128, 128, 128)  # Gray color

        # Update keys and colors after adding undefined types
        veg_keys = sorted(
            vegetation_colors.keys()
        )  # 🔥 Sort to avoid ValueError from BoundaryNorm
        veg_colors = [
            np.array(vegetation_colors[key]) / 255.0 for key in veg_keys
        ]  # Normalize 0-1

        # Create a ListedColormap from the specific vegetation colors
        cmap = ListedColormap(veg_colors, name="vegetation")

        # Create a boundary norm to map pixel values to colors
        boundaries = [key - 0.5 for key in veg_keys] + [veg_keys[-1] + 0.5]
        norm = BoundaryNorm(boundaries, ncolors=len(veg_colors))
    else:
        # If veg_palette is False, use the unique values from the array as the "vegetation keys"
        veg_keys = sorted(unique_values)
        cmap = "viridis"
        norm = None

    # Remove vmin and vmax from imshow() and use norm instead
    im = axes[0].imshow(arr, cmap=cmap, norm=norm)

    # Set the title for the 2D array plot
    axes[0].set_title(f"{title}\n{veg_type_desc}", fontsize=10)

    # Create the colorbar and set integer tick labels
    cbar = fig.colorbar(im, ax=axes[0], orientation="vertical")
    cbar.set_ticks(
        veg_keys
    )  # Set the ticks at the exact positions of the vegetation types
    cbar.set_ticklabels(
        [str(veg_type) for veg_type in veg_keys]
    )  # Set labels as integers

    # Count occurrences of each unique vegetation type
    unique_types, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)

    # Sort values so they appear in ascending order
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

    # Plot the histogram using bar
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
