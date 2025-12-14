import numpy as np

def kmeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X
    """
    # You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

    # ====================== YOUR CODE HERE ======================
    
    # 1. Randomly reorder the indices of the examples
    # np.random.permutation(m) creates a shuffled array from 0 to m-1
    randidx = np.random.permutation(X.shape[0])
    
    # 2. Take the first K examples based on the shuffled indices
    centroids = X[randidx[:K]]
    
    # =============================================================

    return centroids
