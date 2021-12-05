from torch import nn
from tqdm import tqdm
from data import FallData
from cnn.model import CNN, train, test
import torch.optim as optim
import matplotlib.pyplot as plt


def test_training(num_epoch=1000):
    learning_rate = 0.01
    momentum = 0.5
    loss_fn = nn.BCELoss()

    model = CNN(channels=2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    data = FallData(test_size=0.2, batch_size=4, resize=(64, 64))
    losses = []

    for epoch in tqdm(range(1, num_epoch + 1), leave=False, desc="Training"):
        train(model, data.train_loader, optimizer, loss_fn)
        if epoch % (num_epoch / 10) == 0:
            losses.append(test(model, data.test_loader, epoch, loss_fn, verbose=False))

    #  plt.plot(losses)
    #  plt.show()


def main():
    test_training(num_epoch=100)
