import os

def get_vocab_list():
  """
    get_vocab_list reads the fixed vocabulary list in vocab.txt and returns a
    cell array of the words

    def get_vocab_list() reads the fixed vocabulary list in vocab.txt 
    and returns a map of the words in vocab_list.
  """

  # Read the fixed vocabulary list
  script_dir = os.path.dirname(__file__)
  rel_path = "../data/vocab.txt"
  path = os.path.join(script_dir, rel_path)
  vocab_list = {}
  with open(path, "r") as f:
    for line in f:
      line = line.split()
      # Word Index
      idx = int(line[0])
      # Actual Word
      vocab_list[idx] = line[1]

  return vocab_list