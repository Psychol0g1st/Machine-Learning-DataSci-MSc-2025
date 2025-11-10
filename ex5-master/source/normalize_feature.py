import numpy as np

def normalize_features(X, mu=None, sigma=None):
    """FEATURENORMALIZE Normalizes the features in X
      FEATURENORMALIZE(X) returns a normalized version of X where
      the mean value of each feature is 0 and the standard deviation
      is 1. This is often a good preprocessing step to do when
      working with learning algorithms.
    """

    m = X.shape[0]
    if mu is None:
        mu = np.mean(X, axis=0)
    if sigma is None:
        sigma = np.std(X, axis=0, ddof=1)
    # don't change the intercept term
    mu[0] = 0.0
    sigma[0] = 1.0
    for i in range(m):
        X[i, :] = (X[i, :] - mu) / sigma
    return X, mu, sigma
