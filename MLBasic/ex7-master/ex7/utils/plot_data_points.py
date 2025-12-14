import numpy as np
from matplotlib import pyplot as plt


def plot_data_points(X, idx, K):
    """
    Plots data points in X, coloring them so that those with the same
    index assignments in idx have the same color

    plot_data_points(X, idx, K) plots data points in X, coloring them so that those
    with the same index assignments in idx have the same color

    :param X:
    :param idx:
    :param K:

    :return: None
    """

    # Create palette
    R = np.linspace(0, 1 - (1 / (K + 1)), num=K + 1)
    palette = plt.cm.hsv(R)
    colors = palette[idx, :]
    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], s=15, c=colors)
