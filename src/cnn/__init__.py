import torch
import torchvision
from tqdm import tqdm
from data import FallData
from cnn.model import CNN
import torch.nn.functional as F
import torch.optim as optim


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    counter = 0
    for data, target in train_loader:
        data = data.float()
        target = target.to(device)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        counter += 1


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float()
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += float(F.nll_loss(output, target, reduction="sum").item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    tqdm.write(
        f"Epoch {epoch}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)"
    )


learning_rate = 0.01
momentum = 0.5
device = "cpu"


def test_training(model, num_epoch=1000):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    #  data = FallData(test_size=0.2, batch_size=4)
    data = FallData(test_size=0.2, batch_size=4, resize=(64, 64))

    test(model, device, data.train_loader, num_epoch)
    for epoch in tqdm(range(1, num_epoch + 1)):
        train(model, device, data.train_loader, optimizer, epoch)
        if epoch % (num_epoch / 10) == 0:
            test(model, device, data.test_loader, epoch)

    print("\nPerformance on training data:")
    test(model, device, data.train_loader, num_epoch)


def main():
    model = CNN(channels=2).to(device)
    test_training(model, num_epoch=200)

    #  model = Binary_Classifier().to(device)
    #  test_training(model, num_epoch=10)
