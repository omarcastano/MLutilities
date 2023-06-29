from MLutilities.regression.utils import generate_nonlinear_data
from MLutilities.regression.models import PolynomialRegression
from sklearn.model_selection import train_test_split

def plot_poly_reg(degree: int, N: int = 50) -> None:
    """
    helper visualization function to see the results of a fitted polynomial regression model on non-linear data

    Parameters:
    -----------
      degree:
        Degree of the polynomial regression model
      N:
        Number of instances of the generated non-linear data
    """
    X, y = generate_nonlinear_data(N)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = PolynomialRegression(degree=degree)
    model.fit(X_train, y_train)
    model.plot_fitted_model(X_train, y_train, X_test, y_test)
