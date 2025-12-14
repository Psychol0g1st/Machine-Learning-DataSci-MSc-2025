import numpy as np


def project_data(X, U, K):
    """
    Computes the reduced data representation when projecting only
    on to the top k eigenvectors

    Z = project_data(X, U, K) computes the projection of
    the normalized inputs X into the reduced dimensional space spanned by
    the first K columns of U. It returns the projected examples in Z.

    :param X: input dataset
    :param U: eigenvectors of X
    :param K: dimension of projected data

    :return: projection of X into the reduced K-dimensional space
    """

    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    """
    ====================== YOUR CODE HERE ======================
    Instructions: Compute the projection of the data using only the 
                 top K eigenvectors in U (first K columns). For the 
                 i-th example X(i,:), the projection on to the k-th 
                 eigenvector is given as follows:
                       x = X(i, :)';
                       projection_k = x' * U(:, k);
    
    """
    for k in range(K):
        Z[:, k] = X.dot(U[:, k].T)

    return Z
