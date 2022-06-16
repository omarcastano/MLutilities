import pickle

import pkg_resources


def diamonds():
  
  """
  Load Diamonds dataset
  """
  
  path_to_data = pkg_resources.resource_filename(__name__, 'diamonds.pkl')
  a_file = open(path_to_data, "rb")
  
  return pickle.load(a_file)
