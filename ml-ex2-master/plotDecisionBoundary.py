from numpy import size, min, max, array, linspace, zeros
import matplotlib.pyplot as plt
from plotData import plotData
from mapFeature import mapFeature

# PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
# the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with * for the
#   positive examples and o for the negative examples. X is assumed to be
#   a either
#   1) Mx3 matrix, where the first column is an all-ones column for the
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones


def plotDecisionBoundary(theta, X, y):

    # Plot Data
    plotData(X[:, 1:3], y)

    if size(X, 1) <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.xlim([30, 100])
        plt.ylim([30, 100])
    else:
        # Here is the grid range
        u = linspace(-1, 1.5, 50)
        v = linspace(-1, 1.5, 50)

        z = zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid

        for i in range(size(u, 0)):
            for j in range(size(v, 0)):
                z[i, j] = mapFeature(array([u[i]]), array([v[j]])) @ theta


        z = z.T  # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, levels=[0], linewidths=2, colors='c')

def plotDecisionBoundaryScaled(theta, X, y, mean_scores, std_scores):
    # This assumes X contains the intercept term in X[:, 0] and features start at X[:, 1].
    
    # Check if the data is plotted before calling this function to ensure data points
    # are visible with the boundary.
    
    # Plot Data (Assuming X contains the intercept term X[:, 0])
    X_orig = X.copy()
    X_orig[:, 1] = X[:, 1] * std_scores[0] + mean_scores[0]
    X_orig[:, 2] = X[:, 2] * std_scores[1] + mean_scores[1]

    plotData(X_orig[:, 1:3], y)

    # ------------------------------------------------------------------
    # Case 1: Linear Decision Boundary (X has up to 2 features + intercept)
    # ------------------------------------------------------------------
    if X.shape[1] <= 3:
        # 1. Define the plot range using the scaled X[:, 1]
        # X[:, 1] is the first feature (assuming X[:, 0] is the intercept term of ones)
        plot_x_scaled = array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])

        # 2. Calculate the decision boundary line (x2) on the scaled axes
        # Decision boundary: theta[0] + theta[1]*x1_scaled + theta[2]*x2_scaled = 0
        # Solving for x2_scaled: x2_scaled = (-1 / theta[2]) * (theta[1] * x1_scaled + theta[0])
        plot_y_scaled = (-1 / theta[2]) * (theta[1] * plot_x_scaled + theta[0])

        # 3. Unscale the x and y values back to the original scale
        # Original = Scaled * STD + Mean
        
        # The first feature (x1) corresponds to X[:, 1] and std_scores[0], mean_scores[0]
        plot_x_orig = plot_x_scaled * std_scores[0] + mean_scores[0]
        
        # The second feature (x2) corresponds to X[:, 2] and std_scores[1], mean_scores[1]
        plot_y_orig = plot_y_scaled * std_scores[1] + mean_scores[1]

        # 4. Plot the decision boundary using the original values
        plt.plot(plot_x_orig, plot_y_orig)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        
        # Axis limits (based on original data range)
        plt.xlim([30, 100])
        plt.ylim([30, 100])
        
    # ------------------------------------------------------------------
    # Case 2: Non-Linear Decision Boundary (using mapFeature)
    # ------------------------------------------------------------------
    else:
        # 1. Define grid based on the original data range (e.g., [30, 100])
        # We need the original range of the features, which we can estimate from the scaled X
        # by unscaling the min/max values.
        
        # Check if the shape of X is correct before trying to access X[:, 1] and X[:, 2]
        if X.shape[1] < 3:
            raise ValueError("X must have at least 3 columns (intercept + 2 features) for mapFeature case.")
            
        u_orig = linspace(min(X[:, 1]) * std_scores[0] + mean_scores[0], 
                          max(X[:, 1]) * std_scores[0] + mean_scores[0], 50)
        v_orig = linspace(min(X[:, 2]) * std_scores[1] + mean_scores[1], 
                          max(X[:, 2]) * std_scores[1] + mean_scores[1], 50)

        z = zeros((u_orig.size, v_orig.size))
        
        # 2. Evaluate z = theta*x over the original grid, scaling inside the loop
        for i in range(size(u_orig, 0)):
             for j in range(size(v_orig, 0)):
                # Scale the original feature values before mapping
                u_scaled = (u_orig[i] - mean_scores[0]) / std_scores[0]
                v_scaled = (v_orig[j] - mean_scores[1]) / std_scores[1]
                
                # Map the scaled features and calculate z
                # This requires the external 'mapFeature' function
                z[i, j] = mapFeature(array([u_scaled]), array([v_scaled])) @ theta

        z = z.T  # important to transpose z before calling contour

        # 3. Plot z = 0 using the original (u_orig, v_orig) grid points
        plt.contour(u_orig, v_orig, z, levels=[0], linewidths=2, colors='c')
        
        # Also need to set the axis limits appropriately for the original scale
        plt.xlim([30, 100])
        plt.ylim([30, 100])
