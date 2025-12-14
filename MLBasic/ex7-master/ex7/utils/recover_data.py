import numpy as np


def recover_data(Z, U, K):
    """
    Recovers an approximation of the original data when using the
    projected data

    X_rec = recover_data(Z, U, K) recovers an approximation the
    original data that has been reduced to K dimensions. It returns the
    approximate reconstruction in X_rec.

    :param Z: input dataset that previously projected into a K-dimensional space
    :param U: eigenvectors of the original dataset
    :param K: dimension of the projection that resulted Z

    :return: Approximation of the original dataset
    """

    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    """
    ====================== YOUR CODE HERE ======================
    Instructions: Compute the approximation of the data by projecting back
                  onto the original space using the top K eigenvectors in U.
    
                  For the i-th example Z(i,:), the (approximate)
                  recovered data for dimension j is given as follows:
                       v = Z[i,:].T
                       recovered_j = v.T * U[j, 0:K].T
    
                  Notice that U[j, 0:K] is a row vector.
    
    """

    X_rec = Z.dot(U[:,0:K].T)

    return X_rec
