import numpy as np
import matplotlib.pyplot as plt


def np_arr(arr, title):
    """
    Plot 2D numpy arrays
    """
    n_valid = np.sum(~np.isnan(arr))

    plt.imshow(arr, cmap="viridis")
    # plt.title(title)
    # not a type (title hacking)
    plt.suptitle(title, fontsize=12, y=1)
    plt.title(f"Count of non-nan elements: {n_valid}", fontsize=8)

    plt.colorbar()
    plt.show()
