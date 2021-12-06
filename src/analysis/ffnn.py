import numpy as np
from tqdm import tqdm
from data import FallData
from sklearn.neural_network import MLPClassifier


def main():
    for test_size in tqdm(np.arange(0.1, 1, 0.1)):
        data = FallData(test_size=test_size, for_pytorch=False)
        net = MLPClassifier(
            solver="lbfgs",
            alpha=1e-3,
            hidden_layer_sizes=(100, 100, 100, 100),
            random_state=1,
        )

        net.fit(data.X_train, data.y_train)

        z_tilde = net.predict(data.X_test)
        print(
            f"{test_size*100:.0f}%: {sum(z_tilde == data.y_test) / len(z_tilde)*100:.0f}%"
        )
