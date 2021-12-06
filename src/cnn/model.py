import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, channels=1):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(channels, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 3380)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def train(model, train_loader, optimizer, loss_fn):
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


def test(model, test_loader, epoch, loss_fn, verbose=False):
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

    return test_loss, correct / len(test_loader.dataset)
