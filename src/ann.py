#  import torch
import torch.nn as nn

#  from torch.autograd import Variable


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.elu3 = nn.ELU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #  print(x)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.elu3(out)
        out = self.fc4(out)
        return out


#
#  def train(data):
#      count = 0
#      loss_list = []
#      iteration_list = []
#      accuracy_list = []
#      num_epochs = 10
#
#      error = nn.CrossEntropyLoss()
#
#      model = ANNModel(input_dim=data.n_features, hidden_dim=150, output_dim=10)
#
#      criterion = nn.MSELoss()
#      optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
#      for epoch in range(num_epochs):
#          for i, (images, labels) in enumerate(data.train_loader):
#              train = Variable(images.view(-1, 28 * 28))
#              labels = Variable(labels)
#              optimizer.zero_grad()
#              train = train.float()
#              outputs = model(train)
#              loss = error(outputs, labels)
#              loss.backward()
#              optimizer.step()
#              count += 1
#              if count % 50 == 0:
#                  correct = 0
#                  total = 0
#                  for images, labels in data.test_loader:
#                      test = Variable(images.view(-1, 28 * 28))
#                      test = test.float()
#                      outputs = model(test)
#                      predicted = torch.max(outputs.data, 1)[1]
#                      total += len(labels)
#                      correct += (predicted == labels).sum()
#
#                  accuracy = 100 * correct / float(total)
#                  loss_list.append(loss.data)
#                  iteration_list.append(count)
#                  accuracy_list.append(accuracy)
#              if count % 500 == 0:
#                  print(
#                      "Iteration: {}  Loss: {}  Accuracy: {} %".format(
#                          count, loss.data, accuracy
#                      )
#                  )
#
#  def plot():
#      import matplotlib.pyplot as plt
#
#      plt.plot(iteration_list, loss_list)
#      plt.xlabel("Number of iteration")
#      plt.ylabel("Loss")
#      plt.title("ANN: Loss vs Number of iteration")
#      plt.show()
#
#      # visualization accuracy
#      plt.plot(iteration_list, accuracy_list, color="red")
#      plt.xlabel("Number of iteration")
#      plt.ylabel("Accuracy")
#      plt.title("ANN: Accuracy vs Number of iteration")
#      plt.show()
#
#      #  model = ANNModel(input_dim=data.n_features, hidden_dim=150, output_dim=10)
#      #  criterion = nn.MSELoss()
#      #  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#      #  for epoch in range(100):
#      #      inputs = Variable(torch.randn(1, 2))
#      #      target = Variable(torch.randn(1, 1), requires_grad=False)
#      #      optimizer.zero_grad()
#      #      output = model(inputs)
#      #      loss = criterion(output, target)
#      #      loss.backward()
#      #      optimizer.step()
#      #      print("Epoch: {} | Loss: {}".format(epoch, loss.data[0]))
#
#
#  #
#  #  input_dim = 28 * 28
#  #  hidden_dim = 150  # hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
#  #  output_dim = 10
#  #
#  #  # Create ANN
#  #  model = ANNModel(input_dim, hidden_dim, output_dim)
#  #
#  #  # Cross Entropy Loss
#  #  error = nn.CrossEntropyLoss()
#  #
#  #  # SGD Optimizer
#  #  learning_rate = 0.02
#  #  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#  #
#  #
#  #  def train():
#  #      count = 0
#  #      loss_list = []
#  #      iteration_list = []
#  #      accuracy_list = []
#  #      num_epochs = 10
#  #      for epoch in range(num_epochs):
#  #          for i, (images, labels) in enumerate(train_loader):
#  #
#  #              train = Variable(images.view(-1, 28 * 28))
#  #              labels = Variable(labels)
#  #
#  #              # Clear gradients
#  #              optimizer.zero_grad()
#  #
#  #              # Forward propagation
#  #              outputs = model(train)
#  #
#  #              # Calculate softmax and ross entropy loss
#  #              loss = error(outputs, labels)
#  #
#  #              # Calculating gradients
#  #              loss.backward()
#  #
#  #              # Update parameters
#  #              optimizer.step()
#  #
#  #              count += 1
#  #
#  #              if count % 50 == 0:
#  #                  # Calculate Accuracy
#  #                  correct = 0
#  #                  total = 0
#  #                  # Predict test dataset
#  #                  for images, labels in test_loader:
#  #
#  #                      test = Variable(images.view(-1, 28 * 28))
#  #
#  #                      # Forward propagation
#  #                      outputs = model(test)
#  #
#  #                      # Get predictions from the maximum value
#  #                      predicted = torch.max(outputs.data, 1)[1]
#  #
#  #                      # Total number of labels
#  #                      total += len(labels)
#  #
#  #                      # Total correct predictions
#  #                      correct += (predicted == labels).sum()
#  #
#  #                  accuracy = 100 * correct / float(total)
#  #
#  #                  # store loss and iteration
#  #                  loss_list.append(loss.data)
#  #                  iteration_list.append(count)
#  #                  accuracy_list.append(accuracy)
#  #              if count % 500 == 0:
#  #                  # Print Loss
#  #                  print(
#  #                      "Iteration: {}  Loss: {}  Accuracy: {} %".format(
#  #                          count, loss.data, accuracy
#  #                      )
#  #                  )
#  #
#  #
#  #  if __name__ == "__main__":
#  #      train()
