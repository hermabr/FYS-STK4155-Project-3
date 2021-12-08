import config
import numpy as np
from torch import nn
from tqdm import tqdm
from data import FallData
from plot import line_plot
import torch.optim as optim
from cnn.model import CNN, train, test


def train_model(data, num_epoch=config.NUM_EPOCHS):
    learning_rate = config.LEARNING_RATE
    momentum = config.MOMENTUM
    loss_fn = nn.BCELoss()

    model = CNN(channels=config.NUMBER_OF_CHANNELS)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    losses = []

    for epoch in tqdm(range(1, num_epoch + 1), leave=False, desc="Training"):
        train(model, data.train_loader, optimizer, loss_fn)
        if epoch % 10 == 0:
            loss, accuracy, all_targets, all_predictions = test(
                model, data.test_loader, epoch, loss_fn, verbose=False
            )
            losses.append(loss)

            if accuracy == 1:
                break

    return accuracy, losses, all_targets, all_predictions


def main():
    learning_rate = config.LEARNING_RATE
    momentum = config.MOMENTUM
    loss_fn = nn.BCELoss()
    model = CNN(channels=config.NUMBER_OF_CHANNELS)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    NUM_EPOCHS = 100
    SHOULD_TRANSFORM = True
    EPOCHS = list(range(1, NUM_EPOCHS + 1))

    losses = np.empty(NUM_EPOCHS)
    accuracies = np.empty(NUM_EPOCHS)

    data = FallData(
        test_size=config.TEST_SIZE,
        filepath="data/fall_adjusted",
        batch_size=config.BATCH_SIZE,
        resize=config.RESIZE,
        for_pytorch=True,
        transform=SHOULD_TRANSFORM,
    )

    for epoch in tqdm(EPOCHS, leave=False, desc="Training"):
        train(model, data.train_loader, optimizer, loss_fn)
        loss, accuracy, _, _ = test(
            model, data.test_loader, epoch, loss_fn, verbose=False
        )
        losses[epoch - 1] = loss
        accuracies[epoch - 1] = accuracy

    losses = np.reshape(losses, (-1, config.GROUP_SIZE)).mean(axis=1)
    accuracies = np.reshape(accuracies, (-1, config.GROUP_SIZE)).mean(axis=1)

    line_plot(
        "Loss CNN with data augmentation",
        [
            list(
                range(
                    config.GROUP_SIZE, NUM_EPOCHS + config.GROUP_SIZE, config.GROUP_SIZE
                )
            )
        ],
        [losses],
        [""],
        "epoch",
        "loss",
        filename="output/plots/loss_cnn",
    )

    line_plot(
        "Accuracy CNN with data augmentation",
        [
            list(
                range(
                    config.GROUP_SIZE, NUM_EPOCHS + config.GROUP_SIZE, config.GROUP_SIZE
                )
            )
        ],
        [accuracies],
        [""],
        "epoch",
        "accuracy",
        filename="output/plots/accuracy_cnn",
    )
