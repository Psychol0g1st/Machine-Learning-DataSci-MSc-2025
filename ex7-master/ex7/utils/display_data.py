import matplotlib.pyplot as plt
from math import ceil, floor, sqrt
import numpy as np


def display_data(data, ax=None, width=None):
    """
    Display a given list of images on a grid.

    :param data: list of images that can be resized to a square image (e.g. (400, 1) --> (20, 20))
    :param ax: if given, draw data image on this axis
    :param width: width of image instances in the output grid

    :return: plot the given images in a grid
    """
    if not ax:
        ax = plt.gca()

    # Set width automatically if not passed in
    if not width:
        width = round(sqrt(data.shape[1]))

    # Compute rows, cols
    m, n = data.shape
    height = n // width

    # Compute number of items to display
    disp_rows = floor(sqrt(m))
    disp_cols = ceil(m / disp_rows)

    # Padding between images
    pad = 1

    # Setup blank displays
    out_im = np.zeros((pad + disp_rows * (height + pad),
                       pad + disp_cols * (width + pad)),
                      dtype=data.dtype)

    # Copy each data example to out_im
    for i in range(disp_rows):
        for j in range(disp_cols):
            curr_height = pad + i * (height + pad)
            curr_width = pad + j * (width + pad)
            out_im[curr_height:curr_height + height, curr_width:curr_width + width] = \
                np.reshape(data[i * disp_rows + j, :], (height, width)).T

    # Do not show axis
    ax.axis('off')

    # Display image
    ax.imshow(out_im, cmap=plt.cm.gray)
