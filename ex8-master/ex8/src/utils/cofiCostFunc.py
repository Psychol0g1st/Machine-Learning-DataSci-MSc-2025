import numpy as np

def cofiCostFunc(params, Y, R, num_users, num_movies,
                      num_features, lambda_=0.0):
    """
       cofiCostFunc Collaborative filtering cost function.
       cofiCostFunc returns the cost and gradient for the
       collaborative filtering problem.
    """
    # Unfold the U and W matrices from params
    X = params[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # ====================== YOUR CODE HERE ======================
    """
    Instructions: Compute the cost function and gradient for collaborative
                 filtering. Concretely, you should first implement the cost
                 function (without regularization) and make sure it is
                 matches our costs. After that, you should implement the 
                 gradient and use the checkCostFunction routine to check
                 that the gradient is correct. Finally, you should implement
                 regularization.

   Notes: X - num_movies  x num_features matrix of movie features
          Theta - num_users  x num_features matrix of user features
          Y - num_movies x num_users matrix of user ratings of movies
          R - num_movies x num_users matrix, where R(i, j) = 1 if the 
              i-th movie was rated by the j-th user

   You should set the following variables correctly:
          X_grad - num_movies x num_features matrix, containing the 
                   partial derivatives w.r.t. to each element of X
          Theta_grad - num_users x num_features matrix, containing the 
                       partial derivatives w.r.t. to each element of Theta


    """
    # 1. Calculate the prediction error
    # Compute the predicted ratings (X * Theta_transpose)
    predictions = np.dot(X, Theta.T)
    
    # Calculate the difference (Hypothesis - Y)
    # We multiply by R (element-wise) to set errors for unrated movies to 0
    error = (predictions - Y) * R

    # 2. Compute the Cost J
    # Sum of squared errors / 2
    J = 0.5 * np.sum(np.square(error))
    
    # Add Regularization to Cost
    reg_term = (lambda_ / 2) * (np.sum(np.square(Theta)) + np.sum(np.square(X)))
    J = J + reg_term

    # 3. Compute Gradients
    # Gradient for X (Movie features)
    # (Error * Theta) + Regularization
    X_grad = np.dot(error, Theta) + (lambda_ * X)

    # Gradient for Theta (User preferences)
    # (Error_transpose * X) + Regularization
    Theta_grad = np.dot(error.T, X) + (lambda_ * Theta)











    
    
    # =============================================================
    
    grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])
    return J, grad
