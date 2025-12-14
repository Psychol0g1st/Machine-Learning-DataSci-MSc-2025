import numpy as np
import scipy.optimize as optimize

def gaussian_kernel(x1, x2, sigma):
  """
    gaussian_kernel returns a radial basis function kernel between x1 and x2

    def gaussian_kernel(x1, x2) returns a gaussian kernel between x1 and x2
    and returns the value in sim
  """

  # Ensure that x1 and x2 are column vectors
  x1 = x1[:]
  x2 = x2[:]

  # You need to return the following variables correctly.
  sim = 0

  # ====================== YOUR CODE HERE ======================
  # Instructions: Fill in this function to return the similarity between x1
  #               and x2 computed using a Gaussian kernel with bandwidth
  #               sigma
  M = x1.dot(x2.T)
  H1 = np.sum(np.square(np.asmatrix(x1)),1)
  H2 = np.sum(np.square(np.asmatrix(x2)), 1)
  D = H1+H2.T-2*M
  sim = np.exp(-D/(2*sigma**2))


  # =============================================================

  return sim
