import numpy as np

def pca(X):
    """
    Run principal component analysis on the dataset X
    """

    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros((n, n))
    S = np.zeros((n, n))

    # ====================== YOUR CODE HERE ======================
    
    # 1. Compute the Covariance Matrix (Sigma)
    # Formula: Sigma = (1/m) * (X_transpose * X)
    # Note: We assume X has already been mean-normalized (centered)
    # before passing it into this function.
    Sigma = (1 / m) * (X.T @ X)

    # 2. Compute the SVD of the Covariance Matrix
    # np.linalg.svd returns:
    # U: The eigenvectors (n x n)
    # s: The singular values/eigenvalues as a 1D vector (n,)
    # V: The transpose of the eigenvectors (n x n) - not used here
    U, s, V = np.linalg.svd(Sigma)

    # 3. Convert the vector 's' into the diagonal matrix 'S'
    # The prompt initializes S as (n,n), so we conform to that structure.
    S = np.diag(s)
    
    # =============================================================

    return U, S
