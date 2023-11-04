import numpy as np


def generate_nonlinear_data(N: int = 50, err: float = 0.8, rseed: int = 1):
    """
    Generate artificial polynomial data

    Arguments:
    ----------
        N: int, default=50
            Number of samples
        err: float, default=0.0
            Error term
        rseed: int, default=1
            Random seed

    Returns:
    --------
    X: numpy.ndarray
        Input data
    y: numpy.ndarray
        Output data

    Examples:
    ---------
    >>> from MLutilities.regression.utils import generate_nonlinear_data
    >>> X, y = generate_nonlinear_data(N=50)
    >>> print(X.shape)
    (50, 1)
    """

    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1.0 / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)

    return X, y
