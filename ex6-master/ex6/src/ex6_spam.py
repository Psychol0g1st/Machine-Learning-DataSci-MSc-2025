import numpy as np
from sklearn.svm import SVC
from operator import itemgetter
from .utils import \
  load_data, \
  svm_train, \
  svm_predict, \
  linear_kernel, \
  gaussian_kernel, \
  get_vocab_list, \
  read_file, \
  email_features, \
  process_email

def ex6_spam():
  """
    Machine Learning Online Class - Exercise 6 | Spam Classification with SVMs
    
    Instructions
    ------------
    
    This file contains code that helps you get started on the
    exercise. You will need to complete the following functions:
    
       gaussian_kernel.py
       dataset_3_params.py
       process_email.py
       email_features.py
    
    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
  """

  # ==================== Part 1: Email Preprocessing ====================
  # To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
  # to convert each email into a vector of features. In this part, you will
  # implement the preprocessing steps for each email. You should
  # complete the code in process_email.py to produce a word indices vector
  # for a given email.

  print("\nPreprocessing sample email (emailSample1.txt)")

  # Extract Features
  file_contents = read_file('emailSample1.txt')
  word_indices  = process_email(file_contents)

  # Print Stats
  print(f"Word Indices: {word_indices}")
  print()

  # ==================== Part 2: Feature Extraction ====================
  # Now, you will convert each email into a vector of features in R^n. 
  # You should complete the code in email_features.py to produce a feature
  # vector for a given email.

  print('\nExtracting features from sample email (emailSample1.txt)')

  # Extract Features
  file_contents = read_file("emailSample1.txt");
  word_indices  = process_email(file_contents);
  features      = email_features(word_indices);

  # Print Stats
  print(f"Length of feature vector: {len(features)}")
  print(f"Number of non-zero entries: {np.sum(features > 0)}")

  # =========== Part 3: Train Linear SVM for Spam Classification ========
  # In this section, you will train a linear classifier to determine if an
  # email is Spam or Not-Spam.

  # Load the Spam Email dataset
  # You will have X, y in your environment
  (X, y) = load_data.load_spam_train()

  print("\nTraining Linear SVM (Spam Classification)")
  print("(this may take 1 to 2 minutes) ...")

  C = 0.1
  # model = svm_train(X, y, C, linear_kernel)
  # p = svm_predict(model, X)
  svm = SVC(C=C, kernel="linear", tol=1e-3, max_iter=10000)
  svm.set_params(C=C)
  model = svm.fit(X, y.ravel())
  p = np.array([model.predict(X)]).T

  print(f"Training Accuracy: {np.mean(p == y) * 100}")

  # =================== Part 4: Test Spam Classification ================
  # After training the classifier, we can evaluate it on a test set. We have
  # included a test set in spamTest.mat

  # Load the test dataset
  # You will have Xtest, ytest in your environment
  (Xtest, ytest) = load_data.load_spam_test();

  print("\nEvaluating the trained Linear SVM on a test set ...")

  # p = svm_predict(model, Xtest);
  p = np.array([model.predict(Xtest)]).T

  print(f"Test Accuracy: {np.mean(p == ytest) * 100}")


  # ================= Part 5: Top Predictors of Spam ====================
  # Since the model we are training is a linear SVM, we can inspect the
  # weights learned by the model to understand better how it is determining
  # whether an email is spam or not. The following code finds the words with
  # the highest weights in the classifier. Informally, the classifier
  # 'thinks' that these words are the most likely indicators of spam.
  #

  # Sort the weights and obtin the vocabulary list
  to_sort = list(enumerate(model.coef_[0]))
  weight = sorted(to_sort, reverse=True, key=lambda x: x[1])
  vocab_list = get_vocab_list()

  print("\nTop predictors of spam: ")
  for i in range(15):
      print(f" {vocab_list[weight[i][0] + 1]:15} ({weight[i][1]})")
  print("\n")

  # =================== Part 6: Try Your Own Emails =====================
  # Now that you've trained the spam classifier, you can use it on your own
  # emails! In the starter code, we have included spamSample1.txt,
  # spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
  # The following code reads in one of these emails and then uses your 
  # learned SVM classifier to determine whether the email is Spam or 
  # Not Spam

  # Set the file to be read in (change this to spamSample2.txt,
  # emailSample1.txt or emailSample2.txt to see different predictions on
  # different emails types). Try your own emails as well!
  filename = 'emailSample2.txt';

  # Read and predict
  file_contents = read_file(filename)
  word_indices  = process_email(file_contents)
  x             = email_features(word_indices)
  
  # p = svm_predict(model, x)
  p = np.array([model.predict(x.T)]).T

  print(f"\nProcessed {filename}\n\nSpam Classification: {p}")
  print(f"(1 indicates spam, 0 indicates not spam)")