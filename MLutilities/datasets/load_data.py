import pickle


def diamonds():
  
  """
  Load Diamonds dataset
  """
  a_file = open('MLutilities/MLutilities/datasets/diamonds.pkl', "rb")
  
  return pickle.load(a_file)
