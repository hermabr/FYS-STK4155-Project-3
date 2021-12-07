import os
import torch
import random
import imageio
import numpy as np
import pandas as pd
from data.abstract_data import Data
from data.pytorch_dataset import PytorchDataset
from skimage.transform import resize as sk_resize


class FallData(Data):
    def __init__(
        self,
        test_size=None,
        scale_data=True,
        resize=(64, 64),
        filepath="data/fall_adjusted",
        batch_size=64,
        for_pytorch=False,
        for_tensorflow=False,
        transform=True,
    ):
        super().__init__()

        assert test_size is not None, "test_size must be specified"
        assert os.path.exists(filepath), "Filepath does not exist"
        assert os.path.exists(
            os.path.join(filepath, "fall_labels.csv")
        ), "Fall lable-csv do not exist"

        if filepath[-1] != "/" and len(filepath) != 0:
            filepath += "/"

        df = pd.read_csv(os.path.join(filepath, "fall_labels.csv"))
        filenames = df.loc[:, df.columns != "isfall"].values

        X = np.zeros((len(filenames), 2, resize[0], resize[1]), dtype=np.float32)
        y = df.isfall.values
        y = y.astype(np.float32)

        for i, filename in enumerate(filenames):
            for j in range(2):
                motiongram_np = imageio.imread(os.path.join(filepath, filename[j]))
                # only use one channel since the image is grayscale
                motiongram_np = motiongram_np[:, :, 0]
                motiongram_np = motiongram_np.astype(np.double)
                motiongram_np = sk_resize(motiongram_np, resize)
                if scale_data:
                    motiongram_np = motiongram_np / np.max(motiongram_np)
                X[i, j] = motiongram_np

        if for_tensorflow:
            X2 = np.zeros((len(filenames), resize[0], resize[1], 2), dtype=np.float32)
            for i in range(X.shape[0]):
                X2[i] = np.transpose(X[i], (2, 1, 0))
            X = X2
        elif for_pytorch:
            X, y = self.data_to_torch(X, y)
        else:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])

        self.store_data(X, y, test_size)

        if for_tensorflow:
            self.create_train_test_loader_tensorflow(batch_size)
        elif for_pytorch:
            self.create_train_test_loader(batch_size, transform)

    def create_csv(self, filepath, random_seed=42):
        if random_seed:
            random.seed(random_seed)

        all_files = os.listdir(filepath)

        lines = []
        for filter_name in ["fall", "adl"]:
            files = [f for f in all_files if f.startswith(filter_name)]

            for i in range(len(files) // 2):
                files_sorted = sorted(
                    [file for file in files if f"{i + 1:02d}" in file]
                )
                files_sorted.append(str(filter_name == "fall"))
                lines.append(files_sorted)

        with open(os.path.join(filepath, "fall_labels.csv"), "w") as f:
            header = "filename_x,filename_y,isfall\n"
            f.write(header)

            random.shuffle(lines)
            for line in lines:
                f.write(",".join(line) + "\n")

    #  def __len__(self):
    #      return len(self.X_train)
    #
    #  def __getitem__(self, idx):
    #      if self.transform:
    #          X = self.X_train[idx]
    #
    #          transformation = transforms.Compose(
    #              [
    #                  transforms.ToPILImage(),
    #                  transforms.RandomHorizontalFlip(),
    #                  transforms.RandomRotation(20),
    #                  transforms.ToTensor(),
    #                  #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #                  #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #                  transforms.Normalize([0.485, 0.456], [0.229, 0.224]),
    #              ]
    #          )
    #
    #          return transformation(X), self.y_train[idx]
    #          #  for i in range(X.shape[0]):
    #          #      X[i] = transformation(X[i])
    #          #  return X, self.y_train[idx]
    #
    #      return self.X_train[idx], self.y_train[idx]
