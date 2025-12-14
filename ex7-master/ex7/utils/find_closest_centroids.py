import numpy as np

def d(p1, p2):
    return np.sum((p2 - p1) ** 2)

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    """
    # Set K
    K = centroids.shape[0]
    
    # You need to return the following variables correctly.
    idx = np.zeros((X.shape[0]), dtype=np.intp) # intp is safer for indices
    
    # ====================== YOUR CODE HERE ======================
    
    # 1. Expand dimensions to enable broadcasting
    # X shape becomes (m, 1, n)
    # Centroids shape becomes (1, K, n)
    # This allows us to subtract every centroid from every data point at once.
    
    X_expanded = X[:, np.newaxis, :]       # Shape: (m, 1, n)
    centroids_expanded = centroids[np.newaxis, :, :] # Shape: (1, K, n)
    
    # 2. Calculate squared Euclidean distance
    # Resulting shape: (m, K, n) -> sum over n features -> (m, K)
    distances = np.sum((X_expanded - centroids_expanded) ** 2, axis=2)
    
    # 3. Find the index of the minimum distance for each row
    # Resulting shape: (m,)
    idx = np.argmin(distances, axis=1)
    
    # =============================================================

    return idx
