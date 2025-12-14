import numpy as np
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .utils import pca, \
    project_data, \
    recover_data, \
    run_kmeans, \
    kmeans_init_centroids, \
    feature_normalize, \
    draw_line, \
    display_data, \
    plot_data_points


def ex7_pca():
    """
    Machine Learning Online Class
    Exercise 7 | Principle Component Analysis and K-Means Clustering

    Instructions
    ------------

    This file contains code that helps you get started on the
    exercise. You will need to complete the following functions:

       pca.m
       project_data.m
       recover_data.m
       computeCentroids.m
       find_closest_centroids.m
       kmeans_init_centroids.m

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
    """

    """
    ================== Part 1: Load Example Dataset  ===================
     We start this exercise by using a small dataset that is easily to
     visualize

    """
    print('Visualizing example dataset for PCA.\n\n')

    # The following command loads the dataset. You should now have the
    # variable X in your environment
    with open('ex7/data/ex7data1.pkl', 'rb') as fin:
        X = pickle.load(fin)

    # Visualize the example dataset
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'bo')
    plt.axis([0.5, 6.5, 2, 8])
    plt.axis('square')
    plt.pause(0.05)

    print('Program paused. Press enter to continue.\n')
    input()
    """
    =============== Part 2: Principal Component Analysis ===============
     You should now implement PCA, a dimension reduction technique. You
     should complete the code in pca.m
    
    """
    print('\nRunning PCA on example dataset.\n\n')

    # Before running PCA, it is important to first normalize X
    X_norm, mu, sigma = feature_normalize(X)

    # Run PCA
    U, S = pca(X_norm)

    # Compute mu, the mean of the each feature

    # Draw the eigenvectors centered at mean of data. These lines show the
    # directions of maximum variations in the dataset.
    draw_line(mu, mu + 1.5 * S[0, 0] * U[:, 0].T, '--k', linewidth=2)
    draw_line(mu, mu + 1.5 * S[1, 1] * U[:, 1].T, '--k', linewidth=2)
    print('Top eigenvector: \n')
    print(' U[:,0] = {:.6f} {:.6f} \n'.format(U[0, 0], U[1, 0]))
    print('\n(you should expect to see -0.707107 -0.707107)\n')
    plt.show()

    print('Program paused. Press enter to continue.\n')
    input()

    """
    =================== Part 3: Dimension Reduction ===================
     You should now implement the projection step to map the data onto the 
     first k eigenvectors. The code will then plot the data in this reduced 
     dimensional space.  This will show you what the data looks like when 
     using only the corresponding eigenvectors to reconstruct it.
    
     You should complete the code in project_data.py
    """
    plt.close()
    print('\nDimension reduction on example dataset.\n\n')

    # Plot the normalized dataset (returned from pca)
    plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
    plt.axis([-4, 3, -4, 3])
    plt.axis('square')
    # Project the data onto K = 1 dimension
    K = 1
    Z = project_data(X_norm, U, K)
    print('Projection of the first example: {:.6f}\n'.format(Z[0, 0]))
    print('\n(this value should be about 1.481274)\n\n')

    X_rec = recover_data(Z, U, K)
    print('Approximation of the first example: {:.6f} {:.6f}\n'.format(X_rec[0, 0], X_rec[0, 1]))
    print('\n(this value should be about  -1.047419 -1.047419)\n\n')

    # Draw lines connecting the projected points to the original points
    plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
    for i in range(X_norm.shape[0]):
        draw_line(X_norm[i, :], X_rec[i, :], '--k', linewidth=2)
    plt.show()

    print('Program paused. Press enter to continue.\n')
    input()
    """
    =============== Part 4: Loading and Visualizing Face Data =============
     We start the exercise by first loading and visualizing the dataset.
     The following code will load the dataset into your environment
     
    """
    plt.close()
    print('\nLoading face dataset.\n\n')

    # Load Face dataset
    with open('ex7/data/ex7faces.pkl', 'rb') as fin:
        X = pickle.load(fin)

    # Display the first 100 faces in the dataset
    display_data(X[0:100, :])
    plt.title("100 example from faces dataset")
    plt.show()

    print('Program paused. Press enter to continue.\n')
    input()
    """
    =========== Part 5: PCA on Face Data: Eigenfaces  ===================
     Run PCA and visualize the eigenvectors which are in this case eigenfaces
     We display the first 36 eigenfaces.
    
    """
    plt.close()
    print('\nRunning PCA on face dataset.\n(this mght take a minute or two ...)\n\n')

    # Before running PCA, it is important to first normalize X by subtracting
    # the mean value from each feature
    X_norm, mu, sigma = feature_normalize(X)

    # Run PCA
    U, S = pca(X_norm)

    # Visualize the top 36 eigenvectors found
    display_data(U[:, 0:36].T)
    plt.title("top 36 eigenvectors")
    plt.show()

    print('Program paused. Press enter to continue.\n')
    input()

    """
    ============= Part 6: Dimension Reduction for Faces =================
     Project images to the eigen space using the top k eigenvectors 
     If you are applying a machine learning algorithm
    """
    plt.close()
    print('\nDimension reduction for face dataset.\n\n')

    K = 100
    Z = project_data(X_norm, U, K)

    print('The projected data Z has a size of: ')
    print(Z.shape)

    print('Program paused. Press enter to continue.\n')
    input()

    """
    ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
     Project images to the eigen space using the top K eigen vectors and 
     visualize only using those K dimensions
     Compare to the original input, which is also displayed
    """
    print('\nVisualizing the projected (reduced dimension) faces.\n\n')

    K = 100
    X_rec = recover_data(Z, U, K)

    # Display normalized data
    fig, (ax0, ax1) = plt.subplots(1, 2)
    display_data(X_norm[0:100, :], ax=ax0)
    ax0.set_title('Original faces')

    # Display reconstructed data from only k eigenfaces
    display_data(X_rec[0:100, :], ax=ax1)
    ax1.set_title('Recovered faces')

    plt.show()

    print('Program paused. Press enter to continue.\n')
    input()


    """
    === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
     One useful application of PCA is to use it to visualize high-dimensional
     data. In the last K-Means exercise you ran K-Means on 3-dimensional 
     pixel colors of an image. We first visualize this output in 3D, and then
     apply PCA to obtain a visualization in 2D.
    """
    plt.close()

    # Re-load the image from the previous exercise and run K-Means on it
    # For this to work, you need to complete the K-Means assignment first
    A = plt.imread('ex7/data/bird_small.png')

    img_size = A.shape
    X = np.reshape(A, (img_size[0] * img_size[1], 3))
    K = 16
    max_iters = 10
    initial_centroids = kmeans_init_centroids(X, K)
    centroids, idx = run_kmeans(X, initial_centroids, max_iters)

    # Sample 1000 random indexes (since working with all the data is
    # too expensive. If you have a fast computer, you may increase this.
    sel = np.floor(np.random.rand(1000) * X.shape[0]).astype(int) + 1

    # Setup Color Palette
    R = np.linspace(0, 1 - (1 / K), num=K)
    palette = plt.cm.hsv(R / 1.)
    colors = palette[idx[sel], :]

    # Visualize the data and centroid memberships in 3D
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=10, c=colors)
    plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    plt.pause(0.05)

    """
    === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
     Use PCA to project this cloud to 2D for visualization
    """

    # Subtract the mean to use PCA
    X_norm, mu, sigma = feature_normalize(X)

    # PCA and project the data to 2D
    U, S = pca(X_norm)
    Z = project_data(X_norm, U, 2)

    # Plot in 2D
    plt.figure()
    plot_data_points(Z[sel, :], idx[sel], K)
    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    plt.show()
