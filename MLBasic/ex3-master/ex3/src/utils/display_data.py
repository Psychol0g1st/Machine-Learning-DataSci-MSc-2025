import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def display_data(data, fig_label='Data', fig_size=5):
    """
    Display a given list of images.

    :param data: list of images that can be resized to a square image (e.g. (400, 1) --> (20, 20))
    :param fig_label: label of the figure
    :param fig_size: size of the figure in inches

    :return: displays the given images in a grid
    """

    grid_size = int(np.sqrt(np.size(data, 0)))

    if grid_size ** 2 < np.size(data, 0):
        grid_size = grid_size + 1

    fig = plt.figure(figsize=(fig_size, fig_size))
    fig.suptitle(fig_label, fontsize=16)
    # fig.canvas.set_window_title(fig_label)

    gs1 = gridspec.GridSpec(grid_size, grid_size)
    #  set the spacing between axes.
    gs1.update(wspace=0.05, hspace=0.05)

    columns = grid_size
    rows = grid_size

    img_size = int(np.sqrt(np.size(data[0], 0)))

    for i in range(0, columns * rows):
        ax1 = plt.subplot(gs1[i])

        plt.axis('off')

        if i < np.size(data, 0):
            current = data[i]
            img = np.reshape(current, (img_size, img_size)).T
            ax1.imshow(img, cmap='gray')
    plt.show()
