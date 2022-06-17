import pickle
import pandas as pd
import numpy as np
import pkg_resources


def diamonds(load_as:str='dict'):
  
    """
    Returns a dicctionary that loads Diamonds dataset. The dicctionary has 
    the follow keys:
    DESC: a breve description of the dataset and its columns
    data: raw data (rows and columns)
    feature_names: the name of the features/columns

    Argumetns:
        load_as: str. possible options: 'dict', 'list or 'numpy'
            this arguments contros how the raw data is stored.
            * 'dict' : dict like {column -> [values]}
            * 'list' : list like [[column 1],[column 2],[column 3]]
            * 'numpy' : numpy like numpy.array([[column 1],[column 2],[column 3]])
    """

    assert load_as in ['dict', 'list', 'numpy'], "load_as mus be on of the possible options: 'dict', 'list or 'numpy'"

    path_to_data = pkg_resources.resource_filename(__name__, 'diamonds.pkl')
    a_file = open(path_to_data, "rb")
    data = pickle.load(a_file)
  
    if load_as=='dict':
        return {'DESC':data['DESC'], 'data':data.to_dict(orient='list'), 'feature_names':data['feature_names']}
    elif load_as=='list':
        return {'DESC':data['DESC'], 'data':data.to_numpy().tolist(), 'feature_names':data['feature_names']}
    elif load_as=='numpy':
        return {'DESC':data['DESC'], 'data':data.to_numpy(), 'feature_names':data['feature_names']}
