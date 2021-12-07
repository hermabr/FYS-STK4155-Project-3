from torch.utils.data import Dataset
from torchvision.transforms import transforms


class PytorchDataset(Dataset):
    def __init__(self, X, y, is_train=None, transform=False):
        assert not (not transform and is_train is None) or not (
            transform and is_train is not None
        ), "Transform and is_train should not both be specified"

        self.data = X
        self.labels = y

        self.is_train = is_train
        self.transform = transform

    def make_transformation(self, X):
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
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.make_transformation(self.data[idx]), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]
        #  return self.transform(self.data[idx]), self.labels[idx]
