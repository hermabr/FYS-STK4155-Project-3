import os
import torch
import numpy as np
import pandas as pd
from data.abstract_data import Data


class HousePricingData(Data):
    def __init__(
        self, test_size=None, scale_data=True, filepath="data/house", batch_size=64
    ):
        super().__init__()

        assert os.path.isfile(
            os.path.join(filepath, "train.csv")
        ), "train.csv not found"

        if filepath[-1] != "/" and len(filepath) != 0:
            filepath += "/"

        train = pd.read_csv(filepath + "train.csv")
        train.drop("Id", axis=1, inplace=True)

        print(train.columns.values)
        exit()
        X = train.loc[:, train.columns != "SalePrice"].values
        y = train.SalePrice.values

        X, y = self.data_to_torch(X, y)

        self.store_data(X, y, test_size)
        self.create_train_test_loader(batch_size)
        #  print(train[])
        #  missing_data.head(20)
        #  X = train.loc[:, train.columns != "label"].values
        #  y = train.label.values

        #  import matplotlib.pyplot as plt
        #
        #  fig, ax = plt.subplots()
        #  ax.scatter(train["GrLivArea"], train["SalePrice"])
        #  plt.ylabel("SalePrice", fontsize=13)
        #  plt.xlabel("GrLivArea", fontsize=13)
        #  plt.show()
