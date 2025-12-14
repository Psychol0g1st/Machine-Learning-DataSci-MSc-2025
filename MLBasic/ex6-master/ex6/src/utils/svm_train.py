import sys
import numpy as np
import scipy.optimize as optimize

class Model:
  def __init__(self, X, y, kernel_function, b, alphas, w):
    self.X = X
    self.y = y
    self.kernel_function = kernel_function
    self.b = b
    self.alphas = alphas
    self.w = w

def svm_train(X, Y, C, kernel_function, tol = 1e-3, max_passes = 5, sigma = 0):
  """
    svm_train Trains an SVM classifier using a simplified version of the SMO 
    algorithm. 

    def svm_train(X, Y, C, kernelFunction, tol, max_passes, sigma) trains an
    SVM classifier and returns trained model. X is the matrix of training 
    examples.  Each row is a training example, and the jth column holds the 
    jth feature.  Y is a column matrix containing 1 for positive examples 
    and 0 for negative examples.  C is the standard SVM regularization 
    parameter.  tol is a tolerance value used for determining equality of 
    floating point numbers. max_passes controls the number of iterations
    over the dataset (without changes to alpha) before the algorithm quits.
  
    Note: This is a simplified version of the SMO algorithm for training
          SVMs. In practice, if you want to train an SVM classifier, we
          recommend using an optimized package such as:  
  
            LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
            SVMLight (http://svmlight.joachims.org/)
  """

  # Data parameters
  m = np.shape(X)[0]
  n = np.shape(X)[1]

  # Map 0 to -1
  Y = np.where(Y == 0, -1, Y)

  # Variables
  alphas = np.zeros((m, 1))
  b = 0
  E = np.zeros((m, 1))
  passes = 0
  eta = 0
  L = 0
  H = 0

  # Pre-compute the Kernel Matrix since our dataset is small
  # (in practice, optimized SVM packages that handle large datasets
  #  gracefully will _not_ do this)
  # 
  # We have implemented optimized vectorized version of the Kernels here so
  # that the svm training will run faster.
  if kernel_function.__name__ == "linear_kernel":
    # Vectorized computation for the Linear Kernel
    # This is equivalent to computing the kernel on every pair of examples
    K = X @ X.T
  elif kernel_function.__name__ == "gaussian_kernel":
    # Vectorized RBF Kernel
    # This is equivalent to computing the kernel on every pair of examples
    X2 = np.sum(np.square(X), 1)
    K_left = X2
    K_right = X2.T - 2 * (X @ X.T)
    K = K_right
    for i in range(K_left.shape[0]):
        K[i, :] += K_left[i]
    K = np.power(kernel_function(np.array([1]), np.array([0]), sigma), K)
  else:
    # Pre-compute the Kernel Matrix
    # The following can be slow due to the lack of vectorization
    K = np.zeros(m)
    for i in range(m - 1):
      for j in range(m - 1):
        K[i, j] = kernel_function(np.transpose(X[i:]), np.transpose(X[j:]))
        K[j, i] = K[i, j]

  # Train
  print("\nTraining ...", end="")
  dots = 12
  while passes < max_passes:
    num_changed_alphas = 0
    for i in range(m):

      # Calculate Ei = f(x(i)) - y(i) using (2). 
      # E[i] = b + np.sum(X([i, :] * (np.tile(alphas * Y, (1, n)) * X).T) - Y[i]
      E[i] = b + np.sum(alphas * Y * np.array([K[:, i]]).T) - Y[i]

      if (Y[i] * E[i] < -tol and alphas[i] < C) or (Y[i] * E[i] > tol and alphas[i] > 0):
        j = i
        while j == i:
          j = int(np.ceil(m * np.random.rand())) - 1
        # Calculate Ej = f(x(j)) - y(j) using (2).
        E[j] = b + np.sum(alphas * Y * np.array([K[:, j]]).T) - Y[j]

        # Save old alphas
        alpha_i_old = np.copy(alphas[i])
        alpha_j_old = np.copy(alphas[j])

        # Compute L and H by (10) or (11). 
        if Y[i] == Y[j]:
          L = np.maximum(0, alphas[j] + alphas[i] - C)
          H = np.minimum(C, alphas[j] + alphas[i])
        else:
          L = np.maximum(0, alphas[j] - alphas[i])
          H = np.minimum(C, C + alphas[j] - alphas[i])

        if L == H:
            # continue to next i.
            continue

        # Compute eta by (14).
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        if eta >= 0:
            # continue to next i.
            continue

        # Compute and clip new value for alpha j using (12) and (15).
        alphas[j] = alphas[j] - (Y[j] * (E[i] - E[j])) / eta
            
        # Clip
        alphas[j] = np.minimum(H, alphas[j])
        alphas[j] = np.maximum(L, alphas[j])
            
        # Check if change in alpha is significant
        if np.abs(alphas[j] - alpha_j_old) < tol:
            # continue to next i. 
            # replace anyway
            alphas[j] = alpha_j_old
            continue

        # Determine value for alpha i using (16). 
        alphas[i] = alphas[i] + Y[i] * Y[j] * (alpha_j_old - alphas[j])
            
        # Compute b1 and b2 using (17) and (18) respectively.
        b1 = b - E[i] \
              - Y[i] * (alphas[i] - alpha_i_old) * K[i, j].T \
              - Y[j] * (alphas[j] - alpha_j_old) * K[i, j].T
        b2 = b - E[j] \
              - Y[i] * (alphas[i] - alpha_i_old) * K[i, j].T \
              - Y[j] * (alphas[j] - alpha_j_old) * K[j, j].T

        # Compute b by (19). 
        if 0 < alphas[i] and alphas[i] < C:
            b = b1
        elif 0 < alphas[i] and alphas[j] < C:
            b = b2
        else:
            b = (b1 + b2) / 2

        num_changed_alphas = num_changed_alphas + 1

    if num_changed_alphas == 0:
      passes = passes + 1
    else:
      passes = 0

    print(".", end="")
    sys.stdout.flush()
    dots = dots + 1
    if dots > 78:
      dots = 0
      print()

  print(" Done!\n")

  # Save the model
  idx = (alphas > 0).flatten()
  return Model(
    X[idx, :],
    Y[idx],
    kernel_function,
    b,
    alphas[idx],
    ((alphas * Y).T @ X).T
  )