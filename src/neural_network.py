import torch
import torch.nn as nn
from torch.autograd import Variable


def train_ann(data, model):
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    num_epochs = 10

    error = nn.CrossEntropyLoss()

    #  criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data.train_loader):
            train = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
            optimizer.zero_grad()
            train = train.float()
            outputs = model(train)
            loss = error(outputs, labels)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 50 == 0:
                correct = 0
                total = 0
                for images, labels in data.test_loader:
                    test = Variable(images.view(-1, 28 * 28))
                    test = test.float()
                    outputs = model(test)
                    predicted = torch.max(outputs.data, 1)[1]
                    total += len(labels)
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / float(total)
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
            if count % 500 == 0:
                print(
                    "Iteration: {}  Loss: {}  Accuracy: {} %".format(
                        count, loss.data, accuracy
                    )
                )
