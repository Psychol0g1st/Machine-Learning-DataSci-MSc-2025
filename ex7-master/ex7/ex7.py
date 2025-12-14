import numpy as np
from matplotlib import pyplot as plt
import pickle
from .utils import compute_centroids, \
    run_kmeans, \
    find_closest_centroids, \
    kmeans_init_centroids


def ex7():
    """
    Machine Learning Online Class
    Exercise 7 | Principle Component Analysis and K-Means Clustering

    Instructions
    ------------

    This file contains code that helps you get started on the
    exercise. You will need to complete the following functions:

       pca.py
       project_data.py
       recover_data.py
       compute_centroids.py
       find_closest_centroids.py
       kmeans_init_centroids.py

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
    """

    """
    ================= Part 1: Find Closest Centroids ====================
    To help you implement K-Means, we have divided the learning algorithm
    into two functions -- find_closest_centroids and computeCentroids. In this
    part, you shoudl complete the code in the find_closest_centroids function.
    """
    print('Finding closest centroids.\n\n')

    # Load an example dataset that we will be using
    with open('ex7/data/ex7data2.pkl', 'rb') as fin:
        X = pickle.load(fin)

    # Select an initial set of centroids
    K = 3  # 3 Centroids
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    # Find the closest centroids for the examples using the
    # initial_centroids
    idx = find_closest_centroids(X, initial_centroids)

    print('Closest centroids for the first 3 examples: \n')
    print(idx[0:3])
    print('\n(the closest centroids should be 0, 2, 1 respectively)\n')

    print('Program paused. Press enter to continue.\n')
    input()

    """
    ===================== Part 2: Compute Means =========================
    After implementing the closest centroids function, you should now
    complete the computeCentroids function.
    
    """
    print('\nComputing centroids means.\n\n')

    #  Compute means based on the closest centroids found in the previous part.
    centroids = compute_centroids(X, idx, K)

    print('Centroids computed after initial finding of closest centroids: \n')
    print(centroids)
    print('\n(the centroids should be\n')
    print('   [ 2.428301 3.157924 ]\n')
    print('   [ 5.813503 2.633656 ]\n')
    print('   [ 7.119387 3.616684 ]\n)\n')

    print('Program paused. Press enter to continue.\n')
    input()

    """
    =================== Part 3: K-Means Clustering ======================
    After you have completed the two functions computeCentroids and
    find_closest_centroids, you have all the necessary pieces to run the
    kMeans algorithm. In this part, you will run the K-Means algorithm on
    the example dataset we have provided.
    """
    print('\nRunning K-Means clustering on example dataset.\n\n')

    # Load an example dataset
    with open('ex7/data/ex7data2.pkl', 'rb') as fin:
        X = pickle.load(fin)

    # Settings for running K-Means
    K = 3
    max_iters = 10

    """
    For consistency, here we set centroids to specific values
    but in practice you want to generate them automatically, such as by
    settings them to be random examples (as can be seen in
    kmeans_init_centroids).
    """
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

    # Run K-Means algorithm. The 'true' at the end tells our function to plot
    # the progress of K-Means
    centroids, idx = run_kmeans(X, initial_centroids, max_iters, True)
    print('\nK-Means Done.\n\n')

    print('Program paused. Press enter to continue.\n')
    input()

    """
    ============= Part 4: K-Means Clustering on Pixels ===============
    In this exercise, you will use K-Means to compress an image. To do this,
    you will first run K-Means on the colors of the pixels in the image and
    then you will map each pixel on to it's closest centroid.
    
    You should now complete the code in kmeans_init_centroids.py
    """

    print('\nRunning K-Means clustering on pixels from an image.\n\n')

    #  Load an image of a bird
    A = plt.imread('ex7/data/bird_small.png')
    # A = A / 255; # Divide by 255 so that all values are in the range 0 - 1

    # Size of the image
    img_size = A.shape

    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = np.reshape(A, (img_size[0] * img_size[1], 3))

    # Run your K-Means algorithm on this data
    # You should try different values of K and max_iters here
    K = 16
    max_iters = 10

    # When using K-Means, it is important the initialize the centroids
    # randomly.
    # You should complete the code in kmeans_init_centroids.py before proceeding
    initial_centroids = kmeans_init_centroids(X, K)

    # Run K-Means
    [centroids, idx] = run_kmeans(X, initial_centroids, max_iters)

    print('Program paused. Press enter to continue.\n')
    input()

    """
    ================= Part 5: Image Compression ======================
    In this part of the exercise, you will use the clusters of K-Means to
    compress an image. To do this, we first find the closest clusters for
    each example. After that, we 
    """
    print('\nApplying K-Means to compress an image.\n\n')

    # Find closest cluster members
    idx = find_closest_centroids(X, centroids)

    # Essentially, now we have represented the image X as in terms of the
    # indices in idx.

    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by it's index in idx) to the centroid value
    X_recovered = centroids[idx, :]

    # Reshape the recovered image into proper dimensions
    X_recovered = np.reshape(X_recovered, (img_size[0], img_size[1], 3))

    # Display the original image
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(A)
    ax1.set_title('Original')

    # Display compressed image side by side
    ax2.imshow(X_recovered)
    ax2.set_title('Compressed, with {:d} colors.'.format(K))
    plt.show()
