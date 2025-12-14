import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from . import svm_predict

def visualize_boundary(X, y, model, sigma):
  """
    visualize_boundary plots a non-linear decision boundary learned by the SVM
    
    visualize_boundary(X, y, model) plots a non-linear decision 
    boundary learned by the SVM and overlays the data on it
  """

  # Plot the training data on top of the boundary
  pos = np.where(y == 1)
  neg = np.where(y == 0)

  plt.plot(X[pos[0], 0], X[pos[0], 1], 'k+', markersize=7, linewidth=1)
  plt.plot(X[neg[0], 0], X[neg[0], 1], 'yo', markerfacecolor='y', markersize=7)

  # Make classification predictions over a grid of values
  x1plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
  x2plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)

  X1, X2 = np.meshgrid(x1plot, x2plot);
  vals = np.zeros(X1.shape);

  for i in range(X1.shape[1]):
    this_X = np.array([X1[:, i], X2[:, i]]).T
    # vals[:, i] = svm_predict(model, this_X, sigma)
    vals[:, i] = model.predict(this_X)

  # Plot the SVM boundary
  plt.contour(X1, X2, vals);
  plt.show()