import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def visualize_boundary_linear(X, y, model):
  """
    visualize_boundary_linear plots a linear decision boundary learned by the SVM

    def visualize_boundary_linear(X, y, model) plots a linear decision boundary 
    learned by the SVM and overlays the data on it
  """

  # w = model.w
  # b = model.b
  w = model.coef_[0]
  b = model.intercept_[0]
  xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
  yp_left = np.array([w[0]]) * np.array([xp]).T + b
  # yp_right = np.array([w[1]])
  yp_right = np.array([[w[1]]])
  yp = -(np.linalg.solve(yp_right.T, yp_left.T).T)

  pos = np.where(y == 1)
  neg = np.where(y == 0)

  plt.plot(X[pos[0], 0], X[pos[0], 1], 'k+', markersize=7, linewidth=1)
  plt.plot(X[neg[0], 0], X[neg[0], 1], 'yo', markerfacecolor='y', markersize=7)

  plt.plot(xp, yp, 'b-')
  plt.show()
