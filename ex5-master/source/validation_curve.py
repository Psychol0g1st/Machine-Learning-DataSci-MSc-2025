import numpy as np
from .train_linear_regression import train_linear_regression
from .linear_cost_function_reg import linear_cost_function_reg

def validation_curve(X, y, Xval, yval):
    """VALIDATIONCURVE Generate the train and validation errors needed to
    plot a validation curve that we can use to select lambda
      [lambda_vec, error_train, error_val] = ...
          VALIDATIONCURVE(X, y, Xval, yval) returns the train
          and validation errors (in error_train, error_val)
          for different values of lambda. You are given the training set (X,
          y) and validation set (Xval, yval).

    """
    #Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros_like(lambda_vec)
    error_val = np.zeros_like(lambda_vec)

    """====================== YOUR CODE HERE ======================
    Instructions: Fill in this function to return training errors in 
                  error_train and the validation errors in error_val. The 
                  vector lambda_vec contains the different lambda parameters 
                  to use for each calculation of the errors, i.e, 
                  error_train(i), and error_val(i) should give 
                  you the errors obtained after training with 
                  lambda = lambda_vec(i)
    
    Note: You can loop over lambda_vec with the following:
    
          for i = 1:length(lambda_vec)
              lambda = lambda_vec(i);
              % Compute train / val errors when training linear 
              % regression with regularization parameter lambda
              % You should store the result in error_train(i)
              % and error_val(i)
              ....
              
          end
    """


    for i, lambda_test in enumerate(lambda_vec):
        
        # 1. Train the model using the current lambda_test
        #    This gives us the optimal theta for this lambda
        theta = train_linear_regression(X, y, lambda_test)
        
        # 2. Calculate the training error
        #    Use the trained theta
        #    IMPORTANT: Use lambda = 0 for the cost function
        #    (we want the raw error, not the regularized cost)
        train_cost, _ = linear_cost_function_reg(theta, X, y, 0)
        error_train[i] = train_cost
        
        # 3. Calculate the validation error
        #    Use the same trained theta
        #    IMPORTANT: Use lambda = 0 for the cost function
        val_cost, _ = linear_cost_function_reg(theta, Xval, yval, 0)
        error_val[i] = val_cost

    # ============================================================

    return lambda_vec, error_train, error_val
