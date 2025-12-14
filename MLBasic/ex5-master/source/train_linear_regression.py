import scipy.optimize as optimize
import numpy as np
from .linear_cost_function_reg import linear_cost_function_reg

def train_linear_regression(X, y, lambda_):
    """TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    regularization parameter lambda
      [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
      the dataset (X, y) and regularization parameter lambda. Returns the
      trained parameters theta.

    """

    n = X.shape[1]
    initial_theta = np.zeros(n)
    result = optimize.minimize(
        linear_cost_function_reg,
        initial_theta,
        args=(X, y, lambda_),
        method='CG',
        jac=True,
        options={
            'maxiter': 200,
            'disp': False,
        }
    )
    theta = result.x
    return theta