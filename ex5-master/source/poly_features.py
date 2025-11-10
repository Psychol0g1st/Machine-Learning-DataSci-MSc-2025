import numpy as np

def poly_features(X, power=8):
    """
    POLYFEATURES Maps X (1D vector) into the p-th power
      [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
      maps each example into its polynomial features where
      X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

    """
    X_poly = np.zeros((X.shape[0],power))

    """% ====================== YOUR CODE HERE ======================
    Instructions: Given a vector X, return a matrix X_poly where the p-th 
            column of X contains the values of X to the p-th power.
    """
    X_flat = X.flatten()

    # Loop from the first power (1) up to 'power'
    for p in range(1, power + 1):
        # p-1 is the 0-based column index
        X_poly[:, p - 1] = X_flat ** p

    # ============================================================

    # Return the (m x p) matrix
    return X_poly
