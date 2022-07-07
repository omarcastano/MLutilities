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
        load_as: str. possible options: 'dict', 'list, 'numpy' or 'dataframe'
            this argument controls how raw data is return.
            * 'dict' : dict like --> {column : [values]}
            * 'list' : list like --> [[column 1],[column 2],[column 3]]
            * 'numpy1D' : 1D numpy --> like {column: np.array([values])}
            * 'numpy2D': 2D numpy like. shape (rows, columns)
            * 'dataframe: dataframe like pd.DataFrame
            
            
        n: int default(n=10)
            number of instances to randommly sample from the complete dataset.
            If n=-1, the whole dataset is return 
            
    """

    assert load_as in ['dict', 'list', 'numpy', 'dataframe'], "load_as mus be on of the possible options: 'dict', 'list, 'numpy' o 'dataframe'"

    path_to_data = pkg_resources.resource_filename(__name__, 'diamonds.pkl')
    a_file = open(path_to_data, "rb")
    data = pickle.load(a_file)
    
    if n==-1:
        sample = data['data']
    else:
        sample = data['data'].sample(n ,random_state=42)
        
    if load_as=='dict':
        return {'DESC':data['DESC'], 'data':sample.dropna().to_dict(orient='list'), 'feature_names':data['feature_names'].tolist()}
    
    elif load_as=='list':
        return {'DESC':data['DESC'], 'data':sample.dropna().to_numpy().tolist(), 'feature_names':data['feature_names'].tolist()}
    
    elif load_as=='numpy1D':
        
        sample.dropna(inplace=True)
        ds = {}
        for col in sample.columns:
            ds[col] = sample[col].value
            
        return {'DESC':data['DESC'], 'data':ds, 'feature_names':data['feature_names'].tolist()}    

    elif load_as=='numpy2D':
        return {'DESC':data['DESC'], 'data':sample.dropna().to_numpy(), 'feature_names':data['feature_names'].tolist()}
    
    elif load_as=='dataframe':
        return data

