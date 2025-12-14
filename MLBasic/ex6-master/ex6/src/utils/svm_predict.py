import numpy as np
import scipy.optimize as optimize

def svm_predict(model, X, sigma):
  """
    svm_predict returns a vector of predictions using a trained SVM model (svm_train).

    def svm_predict(model, X) returns a vector of predictions using a 
    trained SVM model (svmTrain). X is a mxn matrix where there each 
    example is a row. model is a svm model returned from svmTrain.
    predictions pred is a m x 1 column of predictions of {0, 1} values.
  """

  if X.shape[1] == 1:
    # Examples should be in rows
    X = X.T

  # Dataset 
  m = X.shape[0]
  p = np.zeros((m, 1))
  pred = np.zeros((m, 1))

  if model.kernel_function.__name__ == "linear_kernel":
    # We can use the weights and bias directly if working with the 
    # linear kernel
    p = X * model.w + model.b
  elif model.kernel_function.__name__ == "gaussian_kernel":
    # Vectorized RBF Kernel
    # This is equivalent to computing the kernel on every pair of examples
    X1 = np.array([np.sum(X ** 2, 1)])
    X2 = np.array([np.sum(model.X ** 2, 1).T])
    temp = X @ model.X.T
    K_left = X1
    K_right = X2 - 2 * X @ model.X.T
    K = np.copy(K_right)
    for i in range(K_left.shape[1]):
        K[i, :] += K_left[0][i]
    K = model.kernel_function(np.array([1]), np.array([0]), sigma) ** K;
    K = model.y.T * K
    K = model.alphas.T * K
    p = np.sum(K, 1);
  else:
    # Other Non-linear kernel
    for i in range(m):
      prediction = 0
      for j in range(model.X.shape[0]):
        prediction = prediction + model.alphas[j] * model.y[j] * model.kernel_function(X[i, :].T, model.X[j, :].T);
      p[i] = prediction + model.b;

  # Convert predictions into 0 / 1
  pred = np.copy(p)
  pred = np.where(pred >= 0, 1, pred)
  pred = np.where(pred < 0, 0, pred)

  return pred