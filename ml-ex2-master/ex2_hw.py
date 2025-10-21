#!/usr/bin/env python3
from numpy import where, zeros, mean, loadtxt, array, concatenate, shape, ones
import os
import matplotlib.pyplot as plt
from plotData import plotData
from sigmoid import sigmoid
from costFunctionReg import costFunctionReg
from predict import predict
from scipy.optimize import minimize
from plotDecisionBoundary import plotDecisionBoundary

def ex2():
    # Initialization
    os.system("cls" if os.name == "nt" else "clear")

    # Load Data
    data = loadtxt("ex2data1.txt", delimiter=",")
    X_original = data[:, 0:2]
    y = data[:, 2]

    # ==================== Part 0: Describe data ====================
    mean_X = mean(X_original, axis=0)
    median_X = array([sorted(X_original[:, 0])[len(X_original) // 2], sorted(X_original[:, 1])[len(X_original) // 2]])
    min_X = X_original.min(axis=0)
    max_X = X_original.max(axis=0)
    std_X = X_original.std(axis=0)
    print("Descriptive statistics of the data:")
    print("Mean: {}".format(mean_X))
    print("Median: {}".format(median_X))
    print("Min: {}".format(min_X))
    print("Max: {}".format(max_X))
    print("Standard Deviation: {}".format(std_X))
    
    pos = where(y == 1)
    neg = where(y == 0)
    print("Number of positive examples: {}".format(len(pos[0])))
    print("Number of negative examples: {}".format(len(neg[0])))

    # ==================== Part 1: Plotting ====================
    print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")
    plotData(X_original, y)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.plot([], [], "bo", label="Admitted")
    plt.plot([], [], "r*", label="Not admitted")
    plt.legend()
    plt.show()
    input("Program paused. Press enter to continue.")

    # ============ Part 2: Feature Normalization and Polynomial Expansion ============
    # Normalize features first
    mean_scores = mean(X_original, axis=0)
    std_scores = X_original.std(axis=0)
    X_normalized = (X_original - mean_scores) / std_scores

    # Apply polynomial feature mapping to normalized features
    from mapFeature import mapFeature
    X = mapFeature(X_normalized[:, 0], X_normalized[:, 1])
    
    m, n = shape(X)
    initial_theta = zeros(n)

    # Compute and display initial cost and gradient
    _lambda = 1  # Regularization parameter
    cost, grad = costFunctionReg(initial_theta, X, y, _lambda)

    print("Cost at initial theta (zeros): {:0.3f}".format(cost))
    print("Expected cost (approx): 0.693\n")
    print("Gradient at initial theta (zeros) - first 5 values:")
    for i in range(5):
        print("{:0.4f}".format(grad[i]))
    print("...")

    # ============= Part 3: Optimizing using minimize  =============
    options = {"maxiter": 400}
    result = minimize(fun=costFunctionReg,
                      x0=initial_theta,
                      args=(X, y, _lambda),
                      jac=True,
                      method="TNC",
                      options=options)

    theta = result.x
    cost = result.fun

    # Print theta to screen
    print("Cost at theta found by minimize: {:0.3f}".format(cost))
    print("Expected cost (approx): 0.203")
    print("theta (first 5 values):")
    for i in range(5):
        print("{:0.3f}".format(theta[i]))
    print("...")

    # Plot Boundary (you'll need to update plotDecisionBoundary for polynomial features)
    from plotDecisionBoundary import plotDecisionBoundary
    plotDecisionBoundary(theta, X_normalized, y, mean_scores, std_scores)

    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.plot([], [], "bo", label="Admitted")
    plt.plot([], [], "r*", label="Not admitted")
    plt.legend()
    plt.show()
    input("Program paused. Press enter to continue.")

    # ============== Part 4: Predict and Accuracies ==============
    # Predict for a student with score 45 on exam 1 and score 85 on exam 2
    new_student = array([[45, 85]])
    new_student_normalized = (new_student - mean_scores) / std_scores
    new_student_poly = mapFeature(new_student_normalized[:, 0], new_student_normalized[:, 1])
    
    prob = sigmoid(new_student_poly @ theta)[0]
    print("For a student with scores 45 and 85, we predict an admission probability of {:0.3f}".format(prob))
    print("Expected value: 0.775 +/- 0.002")

    # Compute accuracy on our training set
    p = predict(theta, X)
    print("Train Accuracy: {:0.1f}".format(mean(p == y) * 100))
    print("Expected accuracy (approx): 89.0")

if __name__ == "__main__":
    ex2()
