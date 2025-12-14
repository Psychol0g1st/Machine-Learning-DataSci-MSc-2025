import sys
from ex6.src import ex6
from ex6.src import ex6_spam

def main():
  """
    Machine Learning Class - Exercise 6 - Support Vector Machines
  """
  if len(sys.argv) > 1 and sys.argv[1] == "spam":
    ex6_spam()
  else:
    ex6()


if __name__ == "__main__":
  main()