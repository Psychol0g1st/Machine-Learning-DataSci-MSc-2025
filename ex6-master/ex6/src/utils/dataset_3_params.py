import numpy as np
import scipy.optimize as optimize
from sklearn.svm import SVC
from . import svm_train, svm_predict, gaussian_kernel

def dataset_3_params(X, y, Xval, yval):
  """
    dataset_3_params returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel

    def dataset_3_params(X, y, Xval, yval) returns your choice of C and 
    sigma. You should complete this function to return the optimal C and 
    sigma based on a cross-validation set.
  """

  # You need to return the following variables correctly.
  C = 1;
  sigma = 0.3;

  # ====================== YOUR CODE HERE ======================
  # Instructions: Fill in this function to return the optimal C and sigma
  #               learning parameters found using the cross validation set.
  #               You can use svmPredict to predict the labels on the cross
  #               validation set. For example, 
  #                   predictions = svmPredict(model, Xval);
  #               will return the predictions on the cross validation set.
  #
  #  Note: You can compute the prediction error using 
  #        mean(double(predictions ~= yval))

  par = [0.01,0.03,0.1,0.3,1,3,10,30]
  maxAcc = 0
  bestC = 0
  bestSigma = 0
  for C in par:
    for sigma in par:
      svm = SVC(C=C, kernel=lambda x1, x2: gaussian_kernel(x1, x2, sigma), tol=1e-3, max_iter=10000)
      model = svm.fit(X, y.ravel())
      y_pred = model.predict(Xval)
      acc = np.mean(y_pred == yval[:,0])
      print("model Acc: {} next to C={} and sigma={}".format(acc,C,sigma))
      if acc > maxAcc:
        maxAcc = acc
        bestC = C
        bestSigma = sigma

  # =========================================================================
  C = bestC
  sigma = bestSigma
  print("Final model Acc: {} next to C={} and sigma={}".format(maxAcc, C, sigma))
  return (C, sigma)
