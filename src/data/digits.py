import os
import numpy as np
import pandas as pd
from data.abstract_data import Data


class DigitsData(Data):
    def __init__(
        self, test_size=None, scale_data=True, filepath="data/digits", batch_size=64
    ):
        super().__init__()

        assert os.path.isfile(
            os.path.join(filepath, "train.csv")
        ), "train.csv not found"

        if filepath[-1] != "/" and len(filepath) != 0:
            filepath += "/"

        train = pd.read_csv(filepath + "train.csv")
        X = train.loc[:, train.columns != "label"].values
        y = train.label.values

        if scale_data:
            X = self.scale_data(X)

        X, y = self.data_to_torch(X, y)

        self.store_data(X, y, test_size)

        self.create_train_test_loader(batch_size)

    def scale_data(self, data):
        """Scales the data by scaling to values from 0 to 1

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
        return data
