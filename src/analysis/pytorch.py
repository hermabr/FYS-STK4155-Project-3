import numpy as np
from torch import nn
from tqdm import tqdm
from data import FallData
from cnn.model import CNN, train, test
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(data, num_epoch=150):
    learning_rate = 0.002
    #  learning_rate = 0.002
    #  learning_rate = 0.003
    #  learning_rate = 0.005
    #  learning_rate = 0.01
    momentum = 0.5
    loss_fn = nn.BCELoss()

    model = CNN(channels=2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    losses = []

    for epoch in tqdm(range(1, num_epoch + 1), leave=False, desc="Training"):
        train(model, data.train_loader, optimizer, loss_fn)
        if epoch % 10 == 0:
            #  if epoch == num_epoch - 1:
            loss, accuracy = test(
                model, data.test_loader, epoch, loss_fn, verbose=False
            )
            losses.append(loss)

            if accuracy >= 0.99:
                tqdm.write("BROKE")
                break

    if accuracy != 1:
        print("NJET", accuracy)
    #  loss, accuracy = test(model, data.test_loader, 100, loss_fn, verbose=False)
    return accuracy, losses


def main():
    N = 20
    TEST_SIZE = 0.2
    SHOULD_TRANSFORM = True

    #  test_training(num_epoch=200)
    data = FallData(
        test_size=TEST_SIZE,
        filepath="data/fall_adjusted",
        batch_size=4,
        resize=(64, 64),
        for_pytorch=True,
        transform=SHOULD_TRANSFORM,
    )

    total_accuracy = 0
    all_losses = []
    for _ in range(N):
        data = FallData(
            test_size=TEST_SIZE,
            filepath="data/fall_adjusted",
            batch_size=4,
            resize=(64, 64),
            for_pytorch=True,
            transform=SHOULD_TRANSFORM,
        )

        accuracy, losses = train_model(data, 200)
        total_accuracy += accuracy
        all_losses.append(losses)

    print(f"Acc: {total_accuracy/N}")
    #  for losses in all_losses:
    #      plt.plot(losses)
    #  plt.show()


#  def test_training(num_epoch=1000):
#      learning_rate = 0.01
#      momentum = 0.5
#      loss_fn = nn.BCELoss()
#
#      N = 10
#      SHOULD_TRANSFORM = True
#
#      #  for test_size in np.arange(0.1, 1, 0.1):
#      for test_size in [0.2]:
#          total_accuracy = 0
#          for _ in range(N):
#              model = CNN(channels=2)
#              optimizer = optim.SGD(
#                  model.parameters(), lr=learning_rate, momentum=momentum
#              )
#
#              data = FallData(
#                  test_size=test_size,
#                  filepath="data/fall_adjusted",
#                  batch_size=4,
#                  resize=(64, 64),
#                  for_pytorch=True,
#                  transform=SHOULD_TRANSFORM,
#              )
#
#              for epoch in tqdm(range(1, num_epoch + 1), leave=False, desc="Training"):
#                  train(model, data.train_loader, optimizer, loss_fn)
#                  if epoch % 20 == 0:
#                      loss, accuracy = test(
#                          model, data.test_loader, epoch, loss_fn, verbose=False
#                      )
#                      if accuracy == 1:  # to speed up the process
#                          break
#
#              total_accuracy += accuracy
#              print(f"{test_size*100:.0f}%: {100*accuracy:.2f}%")
#
#          print(f"\nTotal: {test_size*100:.0f}%: {100*total_accuracy/N:.2f}%\n")
#      print(f"Transform: {SHOULD_TRANSFORM}")
#
