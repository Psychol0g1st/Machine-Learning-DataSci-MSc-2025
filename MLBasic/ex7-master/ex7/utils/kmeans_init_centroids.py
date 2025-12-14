import numpy as np


def kmeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X

    centroids = kmeans_init_centroids(X, K) returns K initial centroids to be
    used with the K-Means on the dataset X

    :param X: input dataset
    :param K: number of clusters

    :return: K random points picked from X
    """
    # You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

    """
    ====================== YOUR CODE HERE ======================
    Instructions: You should set centroids to randomly chosen examples from
                  the dataset X
    
    """
    idx = np.random.permutation(X.shape[0])
    centroids = X[idx[0:K],:]

    return centroids
