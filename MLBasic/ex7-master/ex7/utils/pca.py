import numpy as np


def pca(X):
    """
    Run principal component analysis on the dataset X

    [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
    Returns the eigenvectors U, the eigenvalues (on diagonal) in S

    :param X: input dataset

    :return: matrices U and S of eigenvectors and eigenvalues
    """

    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros((n, n))
    S = np.zeros((n, n))

    """
     ====================== YOUR CODE HERE ======================
     Instructions: You should first compute the covariance matrix. Then, you
                   should use the "svd" function to compute the eigenvectors
                   and eigenvalues of the covariance matrix. 
    
     Note: When computing the covariance matrix, remember to divide by m (the
           number of examples).    
    """

    covM = X.T.dot(X)
    U,S,_ = np.linalg.svd(covM)
    S = S * np.eye(S.shape[0])

    return U, S
