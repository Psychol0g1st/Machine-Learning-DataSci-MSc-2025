import numpy as np
import scipy.optimize as optimize

def linear_kernel(x1, x2):
  """
    linear_kernel returns a linear kernel between x1 and x2

    def linearKernel(x1, x2) returns a linear kernel between x1 and x2
  """

  # Ensure that x1 and x2 are column vectors
  x1 = x1[:]
  x2 = x2[:]

  # Compute the kernel
  return x1.T @ x2  # dot product