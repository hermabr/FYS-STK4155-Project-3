import numpy as np
from torch import nn
from tqdm import tqdm
from data import FallData
from cnn.model import CNN, train, test
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(data, num_epoch=100):
    learning_rate = 0.01
    momentum = 0.5
    loss_fn = nn.BCELoss()

    model = CNN(channels=2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    losses = []

    for epoch in tqdm(range(1, num_epoch + 1), leave=False, desc="Training"):
        train(model, data.train_loader, optimizer, loss_fn)
        if epoch % (num_epoch / 5) == 0:
            losses.append(test(model, data.test_loader, epoch, loss_fn, verbose=False))

    loss, accuracy = test(model, data.test_loader, 100, loss_fn, verbose=False)
    return accuracy


def test_training(num_epoch=1000):
    learning_rate = 0.01
    momentum = 0.5
    loss_fn = nn.BCELoss()

    N = 5

    for test_size in np.arange(0.1, 1, 0.1):
        total_accuracy = 0
        for _ in range(N):
            model = CNN(channels=2)
            optimizer = optim.SGD(
                model.parameters(), lr=learning_rate, momentum=momentum
            )
            losses = []

            data = FallData(
                test_size=test_size,
                filepath="data/fall_adjusted",
                batch_size=4,
                resize=(64, 64),
                for_pytorch=True,
            )

            for epoch in tqdm(range(1, num_epoch + 1), leave=False, desc="Training"):
                train(model, data.train_loader, optimizer, loss_fn)
                if epoch % (num_epoch / 5) == 0:
                    losses.append(
                        test(model, data.test_loader, epoch, loss_fn, verbose=False)
                    )

            loss, accuracy = test(model, data.test_loader, 100, loss_fn, verbose=False)
            total_accuracy += accuracy
            print(f"{test_size*100:.0f}%: {100*accuracy:.2f}%")

        print(f"Total: {test_size*100:.0f}%: {100*total_accuracy/N:.2f}%\n")


def main():
    test_training(num_epoch=100)
