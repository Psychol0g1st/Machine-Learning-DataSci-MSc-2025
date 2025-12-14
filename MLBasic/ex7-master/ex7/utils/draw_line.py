from matplotlib import pyplot as plt


def draw_line(p1, p2, *args, **kwargs):
    """
    Draws a line from point p1 to point p2

    draw_line(p1, p2) Draws a line from point p1 to point p2 and holds the
    current figure

    :param p1: start point
    :param p2: end point

    :return: None
    """
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], *args, **kwargs)
