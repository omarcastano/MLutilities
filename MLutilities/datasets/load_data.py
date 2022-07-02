import pickle
import pandas as pd
import numpy as np
import pkg_resources


def diamonds(load_as:str='dict', n=-1):
  
    """
    Returns a dicctionary that loads Diamonds dataset. The dicctionary has 
    the follow keys:
    
    DESC: a breve description of the dataset and its columns
    data: raw data (rows and columns)
    feature_names: the name of the features/columns

    Argumetns:
        load_as: str. possible options: 'dict', 'list or 'numpy'
            this argument controls how raw data is stored.
            * 'dict' : dict like {column -> [values]}
            * 'list' : list like [[column 1],[column 2],[column 3]]
            * 'dataframe: dataframe like pd.DataFrame
            
        n: int default(n=10)
            number of instances to sample from the complete dataset.
            If n=-1, the whole dataset is return 
    """

    assert load_as in ['dict', 'list', 'numpy', 'dataframe'], "load_as mus be on of the possible options: 'dict', 'list, 'numpy' o 'dataframe'"

    path_to_data = pkg_resources.resource_filename(__name__, 'diamonds.pkl')
    a_file = open(path_to_data, "rb")
    data = pickle.load(a_file)
    
    if n==-1:
      n=data['data'].shape[0]
  
    if load_as=='dict':
        return {'DESC':data['DESC'], 'data':data['data'].sample(n, replace=False).to_dict(orient='list'), 'feature_names':data['feature_names'].tolist()}
    elif load_as=='list':
        return {'DESC':data['DESC'], 'data':data['data'].sample(n, replace=False).to_numpy().tolist(), 'feature_names':data['feature_names'].tolist()}
    elif load_as=='numpy':
        return {'DESC':data['DESC'], 'data':data['data'].sample(n, replace=False).to_numpy(), 'feature_names':data['feature_names'].tolist()}
    elif load_as=='dataframe':
        return data
