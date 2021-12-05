from data import FallData
from ffnn import FeedForwardNeuralNetwork
from layers import LinearLayer, LeakyReluLayer


def main():
    data = FallData(test_size=0.2, for_pytorch=False)

    net = FeedForwardNeuralNetwork(
        data.X_train.shape[1],
        [100] * 3,
        classification=True,
        epochs=1000,
        learning_rate=0.01,
        lambda_=0.001,
    )

    net.fit(data.X_train, data.y_train)

    z_tilde = net.predict(data.X_test)
    print(f"{sum(z_tilde == data.y_test) / len(z_tilde) * 100:.0f}")
