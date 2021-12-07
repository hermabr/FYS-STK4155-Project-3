import numpy as np
from tqdm import tqdm
from data import FallData
from analysis import pytorch
from cnn.tensorflow import get_model
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def accuracy_metric(y_true, y_pred):
    return sum(y_pred == y_true) / len(y_pred)


def write_out(name, test_size, accuracy):
    tqdm.write(
        f"{name} test size {test_size*100:.0f}%:\t Accuracy: {accuracy*100:.2f}%"
    )


def main():
    accuracy_texts = []

    for i in range(100):
        #  for i in range(1):
        test_size = 0.2
        #  for test_size in tqdm(np.arange(0.1, 1, 0.1)):
        data = FallData(test_size=test_size)
        data_pytorch = FallData(test_size=test_size, for_pytorch=True)
        data_tensorflow = FallData(test_size=test_size, for_tensorflow=True)

        #  # Logistic regression
        #  logreg = LogisticRegression()
        #  logreg.fit(data.X_train, data.y_train)
        #  z_tilde = logreg.predict(data.X_test)
        accuracy_logistic = 0
        #  accuracy_logistic = accuracy_metric(data.y_test, z_tilde)
        #  write_out("Logistic", test_size, accuracy_logistic)

        #  # Pytorch cnn
        accuracy_pytorch = pytorch.train_model(data_pytorch)
        write_out("Pytorch", test_size, accuracy_pytorch)

        # Tensorflow cnn
        accuracy_tensorflow = 0.1
        #  model = get_model()
        #  model.fit(
        #      data_tensorflow.train_dataset,
        #      epochs=30,
        #      validation_data=data_tensorflow.test_dataset,
        #      verbose=False,
        #  )
        #  _, accuracy_tensorflow = model.evaluate(
        #      data_tensorflow.test_dataset, verbose=False
        #  )
        #  write_out("Tensorflow", test_size, accuracy_tensorflow)

        # Feed forward neural network
        accuracy_ffnn = 0.2
        #  net = MLPClassifier(
        #      solver="lbfgs",
        #      alpha=1e-3,
        #      hidden_layer_sizes=(100, 100, 100, 100),
        #      random_state=1,
        #  )
        #  net.fit(data.X_train, data.y_train)
        #  z_tilde = net.predict(data.X_test)
        #  accuracy_ffnn = accuracy_metric(data.y_test, z_tilde)
        #  write_out("MLP", test_size, accuracy_ffnn)

        #  if (
        #      accuracy_logistic == accuracy_pytorch
        #      and accuracy_logistic == accuracy_ffnn
        #      and accuracy_logistic == 1
        #  ):
        #      print("HA MA GAD")
        #      exit()

        accuracy_texts.append(
            f"{i}){test_size:.1f},{accuracy_logistic*100},{accuracy_pytorch*100},{accuracy_tensorflow*100},{accuracy_ffnn*100}\n"
            #  f"{i}){test_size:.1f},{accuracy_logistic*100},{accuracy_pytorch*100},{accuracy_ffnn*100}\n"
        )

        #  f.write(
        #      f"{test_size*100:.0f},{accuracy_logistic*100:.2f},{accuracy_pytorch*100:.2f},{accuracy_tensorflow*100:.2f},{accuracy_ffnn*100:.2f}\n"
        #  )

    with open("output/data/test_size_performance.csv", "w") as f:
        f.write("test_size,logistic,pytorch cnn,tensorflow cnn,ffnn\n")
        #  f.write("test_size,logistic,pytorch cnn,ffnn\n")
        for line in accuracy_texts:
            f.write(line)
