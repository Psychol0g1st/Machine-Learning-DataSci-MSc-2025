import numpy as np

def project_data(X, U, K):
    """
    Computes the reduced data representation when projecting only
    on to the top k eigenvectors
    """

    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # ====================== YOUR CODE HERE ======================
    
    # 1. Select the first K columns of U (U_reduce)
    # U has shape (n, n). We want (n, K).
    U_reduce = U[:, :K]
    
    # 2. Compute the projection
    # X (m x n) multiplied by U_reduce (n x K) results in Z (m x K)
    Z = X @ U_reduce
    
    # =============================================================

    return Z
