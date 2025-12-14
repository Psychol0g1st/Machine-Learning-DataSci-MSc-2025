import re
from . import get_vocab_list
from nltk.stem.porter import PorterStemmer

def process_email(email_contents):
  """
    process_email preprocesses a the body of an email and
    returns a list of word_indices 

    def process_email(email_contents) preprocesses 
    the body of an email and returns a list of indices of the 
    words contained in the email.
  """

  # Load Vocabulary
  vocab_list = get_vocab_list();

  # Init return value
  word_indices = [];

  # ========================== Preprocess Email ===========================

  # Find the Headers ( \n\n and remove )
  # Uncomment the following lines if you are working with raw emails with the
  # full headers

  # hdrstart = strfind(email_contents, ([char(10) char(10)]));
  # email_contents = email_contents(hdrstart(1):end);

  # Lower case
  email_contents = email_contents.lower()

  # Strip all HTML
  # Looks for any expression that starts with < and ends with > and replace
  # and does not have any < or > in the tag it with a space
  email_contents = re.sub("<[^<>]+>", " ", email_contents)

  # Handle Numbers
  # Look for one or more characters between 0-9
  email_contents = re.sub("[0-9]+", "number", email_contents)

  # Handle URLS
  # Look for strings starting with http:// or https://
  email_contents = re.sub("(http|https)://[^\s]*", "httpaddr", email_contents)

  # Handle Email Addresses
  # Look for strings with @ in the middle
  email_contents = re.sub("[^\s]+@[^\s]+", "emailaddr", email_contents)

  # Handle $ sign
  email_contents = re.sub("[$]+", "dollar", email_contents)

  # ========================== Tokenize Email ===========================

  # Output the email to screen as well
  print("\n==== Processed Email ====\n");

  # Process file
  l = 0;

  # Tokenize and also get rid of any punctuation
  stemmer = PorterStemmer()
  email_contents = re.split(r'[@$/#.-:&\*\+=\[\]?!(){},\'\'\">_<;%\s\n\r\t]+', email_contents)
  for s in email_contents:

    # Remove any non alphanumeric characters
    s = re.sub("[^a-zA-Z0-9]", "", s)

    # Stem the word 
    # (the porter_stemmer sometimes has issues, so we use a try catch block)
    #try:
    s = stemmer.stem(s.strip())
    #except:
    #  s = ""
    #  continue

    # Skip the word if it is too short
    if len(s) < 1:
      continue

    # Look up the word in the dictionary and add to word_indices if
    # found
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to add the index of s to
    #               word_indices if it is in the vocabulary. At this point
    #               of the code, you have a stemmed word from the email in
    #               the variable s. You should look up s in the
    #               vocabulary list (vocabList). If a match exists, you
    #               should add the index of the word to the word_indices
    #               vector. Concretely, if s = 'action', then you should
    #               look up the vocabulary list to find where in vocabList
    #               'action' appears. For example, if vocabList{18} =
    #               'action', then, you should add 18 to the word_indices 
    #               vector (e.g., word_indices = [word_indices ; 18]; ).
    # 
    # Note: vocabList[idx] returns a the word with index idx in the
    #       vocabulary list.
    # 
    # Note: You can use s1 == s2 to compare two strings (s1 and
    #       s2). It will return True only if the two strings are equivalent.
    #



    # =============================================================

    # Print to screen, ensuring that the output lines are not too long
    if (l + len(s)) > 78:
        print()
        l = 0
    print(f"{s} ", end="")
    l = l + len(s) + 1

  # Print footer
  print('\n\n=========================')
  return word_indices