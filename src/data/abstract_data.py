import torch
import numpy as np
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self):
        """Empty initialized for the abstract data class"""
        self.test_size = None

    def scale_data(self, data):
        """Scales the data by scaling to values from 0 to 1, then subtracting the mean

        Parameters
        ----------
            data : np.array
                The data for which to scale

        Returns
        -------
            data : np.array
                A scaled version of the data
        """
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data -= np.mean(data)
        return data

    def store_data(self, X, y, test_size):
        """Stores the data, either as only X, and y, or splitting the X, and z in train/test and saving all

        Parameters
        ----------
            X : np.array
                The X data to save
            y : np.array
                The y data to save
            test_size : float/None
                The test size for which to store the data. None means no test data
        """
        self.test_size = test_size
        if not test_size:
            self._X = X
            self._y = y
        else:
            (
                self._X_train,
                self._X_test,
                self._y_train,
                self._y_test,
            ) = train_test_split(X, y, test_size=test_size)

    def create_train_test_loader(self, batch_size):
        """Creates a train and test loader for the data

        Parameters
        ----------
            batch_size : int
                The batch size for the data loader

        Returns
        -------
            train_loader : torch.utils.data.DataLoader
                The train loader for the data
            test_loader : torch.utils.data.DataLoader
                The test loader for the data
        """
        assert self.X_train is not None, "X_train is None"
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_test, self.y_test),
            batch_size=batch_size,
            shuffle=True,
        )
        self.train_loader = train_loader
        self.test_loader = test_loader

    def data_to_torch(self, X, y):
        """Converts the data to a torch tensor

        Returns
        -------
            X : torch.Tensor
                The X data as a torch tensor
            y : torch.Tensor
                The y data as a torch tensor
        """
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X, y

    def check_property(self, name):
        """Check if a property with a given name is present

        Parameters
        ----------
            name : str
                The name of the property for which to check

        Returns
        -------
            attribute :
                The attribute for the given name

        Raises
        ------
            AttributeError :
                Raises an attribute error if the given attribute does not exist
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(
                f"The data data does not have the attribute '{name[1:]}'. You can only access 'X' and 'y', if there is no test split, and 'X_train', 'y_train', and 'X_test', 'y_test' if there is a test split"
            )

    @property
    def X(self):
        """Get the x-value if it exists

        Returns
        -------
            X : np.array
                Returns the attribute X if it exists
        """
        return self.check_property("_X")

    @property
    def y(self):
        """Get the y-value if it exists

        Returns
        -------
            y : np.array
                Returns the attribute y if it exists
        """
        return self.check_property("_y")

    @property
    def X_train(self):
        """Get the x_train-value if it exists

        Returns
        -------
            x_train : np.array
                Returns the attribute x_train if it exists
        """
        return self.check_property("_X_train")

    @property
    def y_train(self):
        """Get the y_train-value if it exists

        Returns
        -------
            y_train : np.array
                Returns the attribute y_train if it exists
        """
        return self.check_property("_y_train")

    @property
    def X_test(self):
        """Get the x_test-value if it exists

        Returns
        -------
            x_test : np.array
                Returns the attribute x_test if it exists
        """
        return self.check_property("_X_test")

    @property
    def y_test(self):
        """Get the y_test-value if it exists

        Returns
        -------
            y_test : np.array
                Returns the attribute y_test if it exists
        """
        return self.check_property("_y_test")

    @property
    def n_features(self):
        """Returns the number of features in the dataset

        Returns
        -------
            length : int
                The length of the dataset
        """
        if self.test_size:
            return self.X_train.shape[1]
        else:
            return self.X.shape[1]
