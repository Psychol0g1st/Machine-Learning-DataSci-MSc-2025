from ex3.src import ex3, ex3_nn


def main():
    """
    Machine Learning Class - Exercise 3 - Logistic Regression
    """
    #  Part 1
    #  Logistic regression via OneVsAll
    ex3()

    if input('Press ENTER to start the next part. (press [q] to exit here)\n') == 'q':
        print('Exit')
        exit(0)

    #  Part 2
    #  Logistic regression via pre-trained NN
    ex3_nn()


if __name__ == '__main__':
    main()
