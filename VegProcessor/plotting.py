import numpy as np
import matplotlib.pyplot as plt
import logging
import os


logger = logging.getLogger("VegTransition")


def np_arr(arr, title, veg_type_desc="", out_path=None, showplot=False):
    """
    Plot 2D numpy arrays and their histogram using fixed bins (2-26).

    Parameters:
        arr (np.ndarray): 2D numpy array to be plotted.
        title (str): Title for the plot.
        veg_type_desc (str): Description of the input vegetation type to be included in the plot title.
    """
    n_valid = np.sum(~np.isnan(arr))

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Check unique values in the array
    unique_values = np.unique(arr[~np.isnan(arr)])

    # Determine colormap and scaling
    if len(unique_values) == 1:
        cmap = "gray"  # Use black (grayscale) colormap
        unique_value = unique_values[0]
        vmin, vmax = unique_value, unique_value
    else:
        cmap = "viridis"
        vmin, vmax = None, None  # Default scaling

    # Plot the 2D array
    im = axes[0].imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(
        f"{title}\n{veg_type_desc}",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes[0], orientation="vertical")

    # Check if the array is boolean
    if arr.dtype == bool:
        axes[1].text(
            0.5,
            0.5,
            "Boolean Array - Histogram Not Applicable",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Histogram (Empty for Boolean Arrays)", fontsize=10)
        axes[1].axis("off")
    else:
        # Use fixed bins from 2 to 26, corresponding to vegetation types
        flattened = arr[~np.isnan(arr)].flatten()  # Ignore NaN values
        bins = np.arange(2, 27)  # Integers from 2 to 26
        axes[1].hist(flattened, bins=bins, color="blue", alpha=0.7, align="left")
        axes[1].set_title(
            f"Histogram of Array Values\n{veg_type_desc}",
            fontsize=10,
        )
        axes[1].set_xlabel("Vegetation Type")
        axes[1].set_ylabel("Frequency")
        axes[1].set_xticks(bins[:-1])  # Align x-axis ticks with bin centers

    # Adjust layout and show the plots
    plt.tight_layout()

    # Save the plot if outpath is provided
    if out_path:
        # Ensure the directory exists
        os.makedirs(out_path, exist_ok=True)

        # Generate a filename from the title
        sanitized_title = title.replace(" ", "_").replace("\n", "_")
        file_path = os.path.join(out_path, f"{sanitized_title}.png")

        # Save the figure
        fig.savefig(file_path, dpi=300)
        logger.info(f"Saved plot to {file_path}")

    if showplot:
        plt.show()


def sum_changes(
    input_array: np.ndarray,
    output_array: np.ndarray,
    plot_title: str = "Vegetation Type Changes",
    out_path: str = None,
    show_plot: bool = False,
):
    """
    Calculate and plot the changes in vegetation type counts between input and output arrays.

    Parameters:
    - input_array (np.ndarray): The initial vegetation type array.
    - output_array (np.ndarray): The resulting vegetation type array after transitions.
    - plot_title (str): Title for the plot.
    - outpath (str): Directory to save the plot. If None, the plot is not saved.
    - showplot (bool): If True, display the plot after generating it.
    """
    # Ensure the arrays have the same shape
    if input_array.shape != output_array.shape:
        raise ValueError("Input and output arrays must have the same shape.")

    # Count occurrences of each vegetation type in both arrays
    unique_types_input, counts_input = np.unique(
        input_array[~np.isnan(input_array)], return_counts=True
    )
    unique_types_output, counts_output = np.unique(
        output_array[~np.isnan(output_array)], return_counts=True
    )

    # Create a dictionary for easy lookup
    input_counts = dict(zip(unique_types_input, counts_input))
    output_counts = dict(zip(unique_types_output, counts_output))

    # Find all unique vegetation types across both arrays
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

    # Save the plot if outpath is provided
    if out_path:
        os.makedirs(out_path, exist_ok=True)  # Ensure the directory exists
        sanitized_title = plot_title.replace(" ", "_").replace("\n", "_")
        file_path = os.path.join(out_path, f"{sanitized_title}.png")
        plt.savefig(file_path, dpi=300)
        logger.info(f"Saved plot to {file_path}")

    if show_plot:
        plt.show()
        plt.close()
