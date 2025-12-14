import numpy as np


def d(p1, p2):
    return np.sum((p2 - p1) ** 2)


def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    idx = find_closest_centroids(X, centroids) returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])

    :param X: input dataset
    :param centroids: array of cluster centroid points
    :return: array showing datapoints affiliation to a certain cluster
    """
    # Set K
    K = centroids.shape[0]
    # You need to return the following variables correctly.
    idx = np.zeros((X.shape[0]), dtype=np.uint)
    """
    ====================== YOUR CODE HERE ======================
    Instructions: Go over every example, find its closest centroid, and store
                  the index inside idx at the appropriate location.
                  Concretely, idx(i) should contain the index of the centroid
                  closest to example i. Hence, it should be a value in the 
                  range 1..K
    
    Note: You can use a for-loop over the examples to compute this.
    Note: For faster execution you should calculate distances exploiting numpy broadcast capability
    """

    for i in range(0,X.shape[0]):
        idx[i] = np.argmin(np.sum((X[i]-centroids)**2,axis=1))

    return idx
