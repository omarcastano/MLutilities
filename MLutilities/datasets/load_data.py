import pickle

import pkg_resources


def diamonds():
  print( pkg_resources.resource_filename(__name__, 'datasets/diamonds.pkl'))
  """
  Load Diamonds dataset
  """
  a_file = open('datasets/diamonds.pkl', "rb")
  
  return pickle.load(a_file)
