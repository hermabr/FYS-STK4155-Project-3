import os
import random
import imageio
import numpy as np
import pandas as pd
from data.abstract_data import Data
from skimage.transform import resize as sk_resize


class FallData(Data):
    def __init__(
        self,
        test_size=None,
        scale_data=True,
        #  resize=(64, 64),
        resize=(28, 28),
        filepath="data/fall",
        batch_size=64,
    ):
        super().__init__()

        assert os.path.exists(filepath), "Filepath does not exist"
        assert os.path.exists(
            os.path.join(filepath, "fall_labels.csv")
        ), "Fall lable-csv do not exist"

        if filepath[-1] != "/" and len(filepath) != 0:
            filepath += "/"

        df = pd.read_csv(os.path.join(filepath, "fall_labels.csv"))
        filenames = df.loc[:, df.columns != "isfall"].values

        #  X = np.zeros((len(filenames), 2, resize[0], resize[1]), dtype=np.float)
        X = np.zeros((len(filenames), 2, resize[0], resize[1]), dtype=np.float)
        y = df.isfall.values
        y = y.astype(int)
        #  print(y)
        #  exit()
        # read png files from filenames

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

        X, y = self.data_to_torch(X, y)

        self.store_data(X, y, test_size)
        self.create_train_test_loader(batch_size)

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
