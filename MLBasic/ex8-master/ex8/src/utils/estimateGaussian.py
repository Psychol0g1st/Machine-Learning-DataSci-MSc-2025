import numpy as np

def estimateGaussian(X):
    # Useful variables
    m, n = X.shape

    # You should return these values correctly
    mu = np.zeros(n)
    sigma2 = np.zeros(n)

    # ====================== YOUR CODE HERE ======================
    """
    Instructions
    ------------
    Compute the mean of the data and the variances
    In particular, mu[i] should contain the mean of
    the data for the i-th feature and sigma2[i]
    should contain variance of the i-th feature.
   """
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    sigma2 = sigma**2
   
    
    # =============================================================
    return mu, sigma2