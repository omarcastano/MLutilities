import pickle


def diamonds():
  
  """
  Load Diamonds dataset
  """
  a_file = open('./MLutilities/datasets/diamonds.pkl', "rb")
  
  return pickle.load(a_file)
