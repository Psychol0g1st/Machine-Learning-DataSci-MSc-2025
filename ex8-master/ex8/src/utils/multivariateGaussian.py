import numpy as np

def multivariateGaussian(X, mu, Sigma2):
    """
       multivariateGaussian computes the probability 
       density function of the examples X under the multivariate gaussian 
       distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
       treated as the covariance matrix. If Sigma2 is a vector, it is treated
       as the \sigma^2 values of the variances in each dimension (a diagonal
       covariance matrix)
    
    """
    k = mu.size

    # if sigma is given as a diagonal, compute the matrix
    if Sigma2.ndim == 1:
        Sigma2 = np.diag(Sigma2)

    X = X - mu
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(Sigma2) ** (-0.5)\
        * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(Sigma2)) * X, axis=1))
    return p
