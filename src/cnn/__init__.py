import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from data import FallData
from cnn.model import CNN
import torch.optim as optim

import matplotlib.pyplot as plt


def train(model, train_loader, optimizer):
    model.train()
    counter = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        target = target.view_as(output)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        counter += 1


def test(model, test_loader, epoch, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            target = target.view_as(output)
            test_loss += loss_fn(output, target).item()
            pred = output.round()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    if verbose:
        tqdm.write(
            f"Epoch {epoch}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)"
        )

    return test_loss


learning_rate = 0.01
momentum = 0.5
loss_fn = nn.BCELoss()


def test_training(test_sizes=[0.2], num_epoch=1000):

    for i in tqdm(test_sizes, desc="Main loop"):
        model = CNN(channels=2)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        data = FallData(test_size=i, batch_size=4, resize=(64, 64))
        losses = []

        for epoch in tqdm(range(1, num_epoch + 1), leave=False, desc="Training"):
            train(model, data.train_loader, optimizer)
            if epoch % (num_epoch / 10) == 0:
                losses.append(test(model, data.test_loader, epoch, verbose=False))

        plt.plot(losses, label=f"{i:.1f}")
        tqdm.write(f"Performance on test data for {i:.1f}:")
        test(model, data.test_loader, num_epoch, verbose=True)
        tqdm.write("")

    plt.legend()
    plt.show()


def main():
    #  test_training(num_epoch=100)
    #  test_training(range(1, 10), num_epoch=50)

    #  for epoch in tqdm(range(10, 400, 20)):
    #      #  print(epoch)
    #      test_training(test_sizes=[0.3], num_epoch=epoch)

    test_training(np.arange(0.1, 1.0, 0.1), num_epoch=100)
