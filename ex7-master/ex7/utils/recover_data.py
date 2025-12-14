import numpy as np

def recover_data(Z, U, K):
    """
    Recovers an approximation of the original data when using the
    projected data
    """

    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # ====================== YOUR CODE HERE ======================
    
    # 1. Get the top K eigenvectors used for the original projection
    # U_reduce shape: (n, K)
    U_reduce = U[:, :K]
    
    # 2. Project back to the original space
    # To map back, we multiply Z by the transpose of the reduced eigenvectors.
    # Dimensions: (m, K) @ (K, n) -> (m, n)
    X_rec = Z @ U_reduce.T
    
    # =============================================================

    return X_rec
