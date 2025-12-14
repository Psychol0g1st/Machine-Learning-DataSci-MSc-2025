import numpy as np
import imageio
from os import getcwd, path
from PIL import Image


def load_own_image(name):
    """
    Loads a custom 20x20 grayscale image to a (1, 400) vector.

    :param name: name of the image file
    :return: (1, 400) vector of the grayscale image
    """

    print('Loading image:', name)

    file_name = path.join(getcwd(), 'ex3', 'src', 'data', name)
    img = Image.open(file_name).convert('L').resize((20, 20))
    
    # Convert to numpy array
    img = np.asarray(img)

    # reshape 20x20 grayscale image to a vector
    return np.reshape(img.T / 255.0, (1, 400))
