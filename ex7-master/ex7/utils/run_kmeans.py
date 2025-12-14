import numpy as np
from matplotlib import pyplot as plt
from .find_closest_centroids import find_closest_centroids
from .plot_progress_kmeans import plot_progress_kmeans
from .compute_centroids import compute_centroids


def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example.

    centroids, idx = run_kmeans(X, initial_centroids, max_iters, plot_progress)
    runs the K-Means algorithm on data matrix X, where each
    row of X is a single example. It uses initial_centroids used as the
    initial centroids. max_iters specifies the total number of interactions
    of K-Means to execute. plot_progress is a true/false flag that
    indicates if the function should also plot its progress as the
    learning happens. This is set to false by default. run_kmeans returns
    centroids, a Kxn matrix of the computed centroids and idx, a m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])

    :param X: input dataset matrix where clustering is run upon. Each row of X is a single data "point"
    :param initial_centroids: Initial centroids which updated via clustering
    :param max_iters: Max iteration number of the algorithm
    :param plot_progress: if True the cluster centroids update progress plotted on the dataset

    :return: final cluster centroids and an array showing datapoints affiliation to a certain cluster
    """

    # Plot the data if we are plotting progress
    if plot_progress:
        plt.figure()

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m, dtype=np.uint)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print('K-Means iteration {:d}/{:d}...\n'.format(i + 1, max_iters))

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plot_progress_kmeans(X, centroids, previous_centroids, idx, K, i + 1)
            previous_centroids = centroids
            plt.pause(0.5)
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)

    plt.show()
    return centroids, idx
