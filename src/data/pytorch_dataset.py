from torch.utils.data import Dataset
from torchvision.transforms import transforms


class PytorchDataset(Dataset):
    def __init__(self, X, y, is_train=None, transform=False):
        """Initialize the pytorch dataset

        Parameters
        ----------
            X : numpy array
                The data
            y : numpy array
                The labels
            is_train : bool
                Whether the dataset is for training or not
            transform : bool
                Whether to apply transformations to the data
        """
        assert not (not transform and is_train is None) or not (
            transform and is_train is not None
        ), "Transform and is_train should not both be specified"

        self.data = X
        self.labels = y

        self.is_train = is_train
        self.transform = transform

    def make_transformation(self, X):
        """Apply transformations to the data

        Parameters
        ----------
            X : numpy array
                The data

        Returns
        -------
            X : numpy array
                The transformed data
        """
        normalization = transforms.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224])
        if self.is_train:
            transformer = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ToTensor(),
                    normalization,
                ]
            )
        else:
            transformer = transforms.Compose(
                [transforms.ToPILImage(), transforms.ToTensor(), normalization]
            )

        return transformer(X)

    def __len__(self):
        """Return the length of the dataset

        Returns
        -------
            len : int
                The length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Return the item at the given index

        Parameters
        ----------
            idx : int
                The index of the item

        Returns
        -------
            item : tuple
                The item at the given index
        """
        if self.transform:
            return self.make_transformation(self.data[idx]), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]
