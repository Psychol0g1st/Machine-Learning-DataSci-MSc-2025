from ex8.src import ex8_1, ex8_2


def main():
    """
    Machine Learning Class - Exercise 8 - Anomaly detection
    """
    #  Part 1
    #  Anomaly detection
    ex8_1()

    if input('Press ENTER to start the next part. (press [q] to exit here)\n') == 'q':
        print('Exit')
        exit(0)

    #  Part 2
    #  Recommender Systems
    ex8_2()


if __name__ == '__main__':
    main()
