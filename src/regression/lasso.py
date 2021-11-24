from sklearn.linear_model import Lasso as LassoSKLearn
from regression.linear_regression_models import LinearRegression


class Lasso(LinearRegression):
    def __init__(self, degree, lambda_):
        """Constructor for the lasso regression model

        Parameters
        ----------
            degree : int
                The degree for the lasso model
            lambda_ : float
                The value of the penalty term
        """
        super().__init__(degree)
        self.model = LassoSKLearn(lambda_)

    def fit(self, X, z):
        """Fits the data using sklearn lasso

        Parameters
        ----------
            x : np.array
                The x values for which to fit the model
            y : np.array
                The y values for which to fit the model
            z : np.array
                The z values for which to fit the model
        """
        self.model.fit(X, z)

    def predict(self, X):
        """Custom predict method using the sklearn

        Parameters
        ----------
            x : np.array
                The x values for which to predict
            y : np.array
                The y values for which to predict

        Returns
        -------
            z_tilde : np.array
                The predicted z-tilde-values
        """
        z_tilde = self.model.predict(X)
        return z_tilde

    def __repr__(self):
        return "Lasso"
