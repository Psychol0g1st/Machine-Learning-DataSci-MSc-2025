import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def display_data(X, y, fig_label = 'Data', fig_size=5):
  pos = np.where(y == 1)
  neg = np.where(y == 0)

  plt.plot(X[pos[0], 0], X[pos[0], 1], 'k+', markersize=7, linewidth=1)
  plt.plot(X[neg[0], 0], X[neg[0], 1], 'yo', markerfacecolor='y', markersize=7)
  plt.show()