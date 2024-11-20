import numpy as np
import matplotlib.pyplot as plt


def np_arr(arr, title, veg_type_desc=""):
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
        f"{title}\n{veg_type_desc}\nCount of non-NaN elements: {n_valid}",
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
    plt.show()
