import os
def read_file(filename):
  """
    read_file reads a file and returns its entire contents
 
    file_contents = READFILE(filename) reads a file and returns its entire
    contents in file_contents
  """

  # Load File
  script_dir = os.path.dirname(__file__)
  rel_path = "../data/" + filename
  path = os.path.join(script_dir, rel_path)
  with open(path) as f:
    if f.readable():
      file_contents = f.read()
    else:
      file_contents = ""
      print("Unable to open {filename}")

  return file_contents
