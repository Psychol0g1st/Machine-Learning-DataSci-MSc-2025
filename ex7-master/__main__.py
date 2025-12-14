from ex7 import ex7, ex7_pca


def main():
    """
    Machine Learning Class - Exercise 7 - K-means clustering
    """
    #  Part 1
    #  k-means
    # ex7()

    if input('Press ENTER to start the next part. (press [q] to quit)\n') == 'q':
        exit(0)

    #  Part 2
    #  k-means with PCA
    ex7_pca()


if __name__ == "__main__":
    main()
