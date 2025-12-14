import os
import numpy as np
from matplotlib import pyplot
from scipy.io import loadmat
from scipy import optimize
from os.path import join

from .utils import cofiCostFunc,checkCostFunction,loadMovieList,normalizeRatings

def ex8_2():
    """
    Machine Learning Class - Exercise 8 | Part 2: Recommender Systems

    Instructions
    ------------

    This file contains code that helps you get started on the
    exercise. You will need to complete the following functions:

         estimateGaussian.py
         selectThreshold.py
         cofiCostFunc.py

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above
    """

    print()
    print('--------------------------------------')
    print('Recommender Systems')
    print('--------------------------------------')
    print()

    """
    
    =============== Part 1: Loading movie ratings dataset ================
     You will start by loading the movie ratings dataset to understand the
     structure of the data.
    """
    
    # Load data
    data = loadmat(os.path.join(os.getcwd(), 'ex8/src/data/ex8_movies.mat'))

    Y, R = data['Y'], data['R']

    # Y is a 1682x943 matrix, containing ratings (1-5) of 
    # 1682 movies on 943 users

    # R is a 1682x943 matrix, where R(i,j) = 1 
    # if and only if user j gave a rating to movie i

    # From the matrix, we can compute statistics like average rating.
    print('Average rating for movie 1 (Toy Story): %f / 5' %
          np.mean(Y[0, R[0, :]]))

    # We can "visualize" the ratings matrix by plotting it with imshow
    pyplot.figure(figsize=(8, 8))
    pyplot.imshow(Y)
    pyplot.ylabel('Movies')
    pyplot.xlabel('Users')
    pyplot.grid(False)
    pyplot.show()
    """
    ============ Part 2: Collaborative Filtering Cost Function ===========
     You will now implement the cost function for collaborative filtering.
     To help you debug your cost function, we have included set of weights
     that we trained on that. Specifically, you should complete the code in 
     cofiCostFunc.m to return J.
    """
    input('\nPress ENTER to start Filtering Cost Function')
    print('Filtering Cost Function...')

    #  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    data = loadmat(os.path.join(os.getcwd(), 'ex8/src/data/ex8_movieParams.mat'))

    X, Theta, num_users, num_movies, num_features = data['X'],\
            data['Theta'], data['num_users'], data['num_movies'], data['num_features']

    #  Reduce the data set size so that this runs faster
    num_users = 4
    num_movies = 5
    num_features = 3

    X = X[:num_movies, :num_features]
    Theta = Theta[:num_users, :num_features]
    Y = Y[:num_movies, 0:num_users]
    R = R[:num_movies, 0:num_users]
    #  Evaluate cost function
    J, _ = cofiCostFunc(np.concatenate([X.ravel(), Theta.ravel()]),
                        Y, R, num_users, num_movies, num_features)

    print('Cost at loaded parameters:  %.2f \n(this value should be about 22.22)' % J)

    """
    ============== Part 3: Collaborative Filtering Gradient ==============
     Once your cost function matches up with ours, you should now implement 
     the collaborative filtering gradient function. Specifically, you should 
     complete the code in cofiCostFunc.m to return the grad argument.
    """
    input('\nPress ENTER to start Filtering Gradient')
    print('Filtering Gradient...')

    #  Check gradients by running checkcostFunction
    checkCostFunction(cofiCostFunc)

    """
    ========= Part 4: Collaborative Filtering Cost Regularization ========
     Now, you should implement regularization for the cost function for 
     collaborative filtering. You can implement it by adding the cost of
     regularization to the original cost computation.
    """
    input('\nPress ENTER to start Collaborative Filtering Cost Regularization')
    print('Collaborative Filtering Cost Regularization...')
    #  Evaluate cost function
    J, _ = cofiCostFunc(np.concatenate([X.ravel(), Theta.ravel()]),
                        Y, R, num_users, num_movies, num_features, 1.5)

    print('Cost at loaded parameters (lambda = 1.5): %.2f' % J)
    print('              (this value should be about 31.34)')

    """
    ======= Part 5: Collaborative Filtering Gradient Regularization ======
     Once your cost matches up with ours, you should proceed to implement 
     regularization for the gradient. 
    """
    input('\nPress ENTER to start Collaborative Filtering Gradient Regularization')
    print('Collaborative Filtering Gradient Regularization...')
    #  Check gradients by running checkCostFunction
    checkCostFunction(cofiCostFunc, 1.5)
    """
    ============== Part 6: Entering ratings for a new user ===============
    Before we will train the collaborative filtering model, we will first
    add ratings that correspond to a new user that we just observed. This
    part of the code will also allow you to put in your own ratings for the
    movies in our dataset!
    """
    input('\nPress ENTER to start Entering ratings for a new user')
    print('Entering ratings for a new user...')
   
    movieList = loadMovieList()
    n_m = len(movieList)

    #  Initialize my ratings
    my_ratings = np.zeros(n_m)

    # Check the file movie_idx.txt for id of each movie in our dataset
    # For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
    # Note that the index here is ID-1, since we start index from 0.
    my_ratings[0] = 4

    # Or suppose did not enjoy Silence of the Lambs (1991), you can set
    my_ratings[97] = 2

    # We have selected a few movies we liked / did not like and the ratings we
    # gave are as follows:
    my_ratings[6] = 3
    my_ratings[11]= 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5

    print('New user ratings:')
    print('-----------------')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print('Rated %d stars: %s' % (my_ratings[i], movieList[i]))
    
    """
    ================== Part 7: Learning Movie Ratings ====================
     Now, you will train the collaborative filtering model on a movie rating 
     dataset of 1682 movies and 943 users
    """
    input('\nPress ENTER to start Learning Movie Ratings')
    print('Learning Movie Ratings...')
    #  Load data
    data = loadmat(os.path.join(os.getcwd(), 'ex8/src/data/ex8_movies.mat'))
    Y, R = data['Y'], data['R']

    #  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
    #  943 users

    #  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
    #  rating to movie i

    #  Add our own ratings to the data matrix
    Y = np.hstack([my_ratings[:, None], Y])
    R = np.hstack([(my_ratings > 0)[:, None], R])

    #  Normalize Ratings
    Ynorm, Ymean = normalizeRatings(Y, R)

    #  Useful Values
    num_movies, num_users = Y.shape
    num_features = 10

    # Set Initial Parameters (Theta, X)
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_users, num_features)

    initial_parameters = np.concatenate([X.ravel(), Theta.ravel()])

    # Set options for scipy.optimize.minimize
    options = {'maxiter': 100}

    # Set Regularization
    lambda_ = 10
    res = optimize.minimize(lambda x: cofiCostFunc(x, Ynorm, R, num_users,
                                                   num_movies, num_features, lambda_),
                            initial_parameters,
                            method='TNC',
                            jac=True,
                            options=options)
    theta = res.x

    # Unfold the returned theta back into U and W
    X = theta[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = theta[num_movies*num_features:].reshape(num_users, num_features)

    print('Recommender system learning completed.')

    """
    ================== Part 8: Recommendation for you ====================
     After training the model, you can now make recommendations by computing
     the predictions matrix.
    """
    input('\nPress ENTER to start Recommendation for you')
    print('Recommendation for you...')

    p = np.dot(X, Theta.T)
    my_predictions = p[:, 0] + Ymean

    movieList = loadMovieList()

    ix = np.argsort(my_predictions)[::-1]

    print('Top recommendations for you:')
    print('-----------------------------')
    for i in range(10):
        j = ix[i]
        print('Predicting rating %.1f for movie %s' % (my_predictions[j], movieList[j]))

    print('\nOriginal ratings provided:')
    print('--------------------------')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print('Rated %d for %s' % (my_ratings[i], movieList[i]))