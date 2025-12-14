import scipy.io
from os import getcwd, path

def load_data1():
  file_name = path.join(getcwd(), "ex6", "src", "data", "ex6data1")
  data = scipy.io.loadmat(file_name)
  return (data['X'], data['y'])

def load_data2():
  file_name = path.join(getcwd(), "ex6", "src", "data", "ex6data2")
  data = scipy.io.loadmat(file_name)
  return (data['X'], data['y'])

def load_data3():
  file_name = path.join(getcwd(), "ex6", "src", "data", "ex6data3")
  data = scipy.io.loadmat(file_name)
  return (data['X'], data['Xval'], data['y'], data['yval'])

def load_spam_test():
  file_name = path.join(getcwd(), "ex6", "src", "data", "spamTest")
  data = scipy.io.loadmat(file_name)
  return (data['Xtest'], data['ytest'])

def load_spam_train():
  file_name = path.join(getcwd(), "ex6", "src", "data", "spamTrain")
  data = scipy.io.loadmat(file_name)
  return (data['X'], data['y'])