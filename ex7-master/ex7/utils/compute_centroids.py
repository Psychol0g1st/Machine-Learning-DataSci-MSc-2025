import numpy as np

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.
    """
    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))

    # ====================== YOUR CODE HERE ======================
    
    # Iterate over every centroid
    for i in range(K):
        # 1. Create a boolean mask where idx equals the current centroid i.
        # Note: We use idx.flatten() to ensure the array is 1D (shape (m,)) 
        # so it works correctly as a mask for the rows of X.
        indices = (idx.flatten() == i)
        
        # 2. Select all rows from X that belong to this centroid
        points_in_cluster = X[indices]
        
        # 3. Compute the mean of these points along the column axis (axis 0)
        # and assign it to the ith row of centroids.
        # We add a check to avoid division by zero if a cluster has no points.
        if len(points_in_cluster) > 0:
            centroids[i] = np.mean(points_in_cluster, axis=0)
            
    # =============================================================

    return centroids
