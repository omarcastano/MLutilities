import pickle


def diamonds():
  
  """
  Load Diamonds dataset
  """
  a_file = open('datasets/diamonds.pkl', "rb")
  
  return pickle.load(a_file)
