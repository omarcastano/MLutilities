import pickle
import pkg_resources

DATASETS = ["penguins", "diamonds", "diamonds_noisy", "adults", "cars", "cars_noisy", "imdb"]


def data_format(data: pickle, load_as: str, n: int):
    """
    Return data in different formats (Dataframe, dict, 2D numpy array, List)

     Argumetns:
         data: raw data within a dictionary that must have the following keys:
                DESC: a breve description of the dataset and its columns
                data: raw data within pandas dataframe (rows and columns)
                image: Illustrative image of the problem (can be none)
                feature_names: the name of the features/columns

        load_as: str. possible options: 'dict', 'list, 'numpy1D', 'numpy2D' or 'dataframe'
            this argument controls how raw data is return.
            * 'dict' : dict like --> {column : [values]}
            * 'list' : list like --> [[column 1],[column 2],[column 3]]
            * 'numpy1D' : 1D numpy --> like {column: np.array([values])}
            * 'numpy2D': 2D numpy like. shape (rows, columns)
            * 'dataframe: dataframe like pd.DataFrame

        n: int default(n=10)
            number of instances to randommly sample from the complete dataset.
            If n=-1, the whole dataset is return. If load_as='datarame', n will be
            ignore and the hole dataset is return
    """

    if n == -1:
        sample = data["data"]
    else:
        sample = data["data"].sample(n, random_state=42)

    if load_as == "dict":
        return {
            "DESC": data["DESC"],
            "image": data["image"],
            "data": sample.dropna().to_dict(orient="list"),
            "feature_names": data["feature_names"],
        }

    elif load_as == "list":
        return {
            "DESC": data["DESC"],
            "image": data["image"],
            "data": sample.dropna().to_numpy().tolist(),
            "feature_names": data["feature_names"],
        }

    elif load_as == "numpy1D":
        sample.dropna(inplace=True)
        ds = {}
        for col in sample.columns:
            ds[col] = sample[col].values

        return {
            "DESC": data["DESC"],
            "image": data["image"],
            "data": ds,
            "feature_names": data["feature_names"],
        }

    elif load_as == "numpy2D":
        return {
            "DESC": data["DESC"],
            "image": data["image"],
            "data": sample.dropna().to_numpy(),
            "feature_names": data["feature_names"],
        }

    elif load_as == "dataframe":
        return data


def load_dataset(data_set: str, load_as: str = "dict", n=-1):
    """
    Returns a dicctionary with the selected dataset. The dicctionary has
    the follow keys:

    DESC: a breve description of the dataset and its columns
    data: raw data (rows and columns)
    image: Image o penguin body parts
    feature_names: the name of the features/columns

    Argumetns:
        dataset: str, possible options: "penguins", "diamonds", "adults", "cars", "cars_nulls"
            name of the dataset to load
        load_as: str. possible options: 'dict', 'list, 'numpy' or 'dataframe'
            this argument controls how raw data is return.
            * 'dict' : dict like --> {column : [values]}
            * 'list' : list like --> [[column 1],[column 2],[column 3]]
            * 'numpy1D' : 1D numpy --> like {column: np.array([values])}
            * 'numpy2D': 2D numpy like. shape (rows, columns)
            * 'dataframe: dataframe like pd.DataFrame
        n: int, default=10
            number of instances to randommly sample from the complete dataset.
            If n=-1, the whole dataset is return
    """

    if data_set not in DATASETS:
        raise Exception(f"This is not an avaliable dataset, the avaliable dataset are {DATASETS}")

    if load_as not in ["dict", "list", "numpy1D", "numpy2D", "dataframe"]:
        raise Exception("load_as mus be on of the possible options: 'dict', 'list, 'numpy1D', 'numpy2D' o 'dataframe'")

    path_to_data = pkg_resources.resource_filename(__name__, f"data/{data_set}.pkl")

    with open(path_to_data, "rb") as f:
        data = pickle.load(f)

    return data_format(data, load_as, n)
