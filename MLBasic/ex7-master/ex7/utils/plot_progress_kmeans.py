from matplotlib import pyplot as plt

from .draw_line import draw_line
from .plot_data_points import plot_data_points


def plot_progress_kmeans(X, centroids, previous, idx, K, i):
    """
    Helper function that displays the progress of
    k-Means as it is running. It is intended for use only with 2D data.

    plot_progress_kmeans(X, centroids, previous, idx, K, i) plots the data
    points with colors assigned to each centroid. With the previous
    centroids, it also plots a line between the previous locations and
    current locations of the centroids.

    :param X: input dataset
    :param centroids: currently calculated centroids
    :param previous: previously calculated centroids
    :param idx: array showing datapoints affiliation to a certain cluster
    :param K: number of cluster centroids
    :param i: actual iteration number

    :return: None
    """

    # Plot the examples
    plot_data_points(X, idx, K)

    # Plot the centroids as black x's
    plt.plot(centroids[:, 0], centroids[:, 1],
             'x', markeredgecolor='k', markersize=10, linewidth=3)

    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous[j, :])

    # Title
    plt.title('Iteration number {:d}'.format(i))
