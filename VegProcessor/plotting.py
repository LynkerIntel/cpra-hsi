import numpy as np
import matplotlib.pyplot as plt


# def np_arr(arr, title):
#     """
#     Plot 2D numpy arrays
#     """
#     n_valid = np.sum(~np.isnan(arr))

#     plt.imshow(arr, cmap="viridis")
#     # plt.title(title)
#     # not a type (title hacking)
#     plt.suptitle(title, fontsize=12, y=1)
#     plt.title(f"Count of non-nan elements: {n_valid}", fontsize=8)

#     plt.colorbar()
#     plt.show()


def np_arr(arr, title):
    """
    Plot 2D numpy arrays and their histogram using fixed bins (2-26).

    Parameters:
        arr (np.ndarray): 2D numpy array to be plotted.
        title (str): Title for the plot.
    """
    n_valid = np.sum(~np.isnan(arr))

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the 2D array
    im = axes[0].imshow(arr, cmap="viridis")
    axes[0].set_title(f"{title}\nCount of non-NaN elements: {n_valid}", fontsize=10)
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
        # Use fixed bins from 2 to 26
        flattened = arr[~np.isnan(arr)].flatten()  # Ignore NaN values
        bins = np.arange(2, 27)  # Integers from 2 to 26
        axes[1].hist(flattened, bins=bins, color="blue", alpha=0.7, align="left")
        axes[1].set_title("Histogram of Array Values", fontsize=10)
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Frequency")
        axes[1].set_xticks(bins[:-1])  # Align x-axis ticks with bin centers

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
