import numpy as np
from .utils import load_data, load_weights, load_own_image, display_data, predict_nn


def ex3_nn():
    """
    Machine Learning Class - Exercise 3 | Part 2: Neural Networks

    Instructions
    ------------
    This file contains code that helps you get started on the
    linear exercise. You will need to complete the following function:

        - predict_nn.py

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
    """

    print()
    print('--------------------------------------')
    print('Classification via pre-trained NN')
    print('--------------------------------------')
    print()

    """
    =========== Part 1: Loading and Visualizing Data =============
    We start the exercise by first loading and visualizing the dataset.
    You will be working with a dataset that contains handwritten digits.
    """
    (X, y) = load_data()
    m = np.size(X, 0)

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[0:100], :]

    display_data(sel, 'Randomly selected training images')

    """
    ================ Part 2: Loading Pameters ================
    In this part of the exercise, we load some pre-initialized 
    neural network parameters.
    """
    (theta1, theta2) = load_weights()

    """
    ================= Part 3: Implement Predict =================
    After training the neural network, we would like to use it to predict
    the labels. You will now implement the "predict" function to use the
    neural network to predict the labels of the training set. This lets
    you compute the training set accuracy.
    """
    pred = predict_nn(theta1, theta2, X)
    print('Predicted classes for X')
    print(np.reshape(pred, (10, 500)))
    print('Training Set Accuracy: ', np.mean((pred == y)) * 100, '%')

    print('Visualising thetas for each class')

    thetas = []
    for i in range(0, np.size(theta1, 0)):
        thetas.append(theta1[i][1:].T)
    display_data(thetas, 'Theta1')

    thetas = []
    for i in range(0, np.size(theta2, 0)):
        thetas.append(theta2[i][1:].T)
    display_data(thetas, 'Theta2')

    """
    ================ Part 4: Test own image ======================
    You can try editing this image (/src/data/jordan5.png) in paint to see how the model deals with it.
    """
    test = load_own_image('jordan5.png')
    test_pred = predict_nn(theta1, theta2, test)

    print('Predicting own image:', test_pred[0] % 10)
    test_img_label = 'Own image\nPredicted: %d' % (test_pred[0] % 10)
    display_data(test, test_img_label)


    """
    ================ Part 5: Test one by one =====================
    To give you an idea of the network's output, you can also run
    through the examples one at a time to see what it is predicting
    """
    rp = np.random.permutation(m)

    print('Displaying examples one-by-one')
    for i in range(0, m):
        example = np.reshape(X[rp[i]], (1, 400))
        pred = predict_nn(theta1, theta2, example)

        pred_num = pred[0] % 10
        print('Example prediction:', pred_num)
        fig_title = 'Training example %d / %d\nPredicted: %d, Label: %d' % (rp[i], np.size(rp, 0), pred_num, y[rp[i]])
        display_data(example, fig_title)

        if input('Press ENTER to show next random example (press [q] to exit)\n') == 'q':
            print('Exit')
            exit(0)
