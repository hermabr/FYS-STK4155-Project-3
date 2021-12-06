import numpy as np
from data import FallData
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def main():
    accuracies = []
    for test_size in np.arange(0.1, 1, 0.1):
        data = FallData(test_size=test_size, for_pytorch=False)

        logreg = LogisticRegression()
        logreg.fit(data.X_train, data.y_train)
        z_tilde = logreg.predict(data.X_test)

        accuracies.append(sum(z_tilde == data.y_test) / len(z_tilde))
        print(
            f"{test_size*100:.0f}%: {sum(z_tilde == data.y_test) / len(z_tilde)*100:.0f}% \t {f1_score(data.y_test, z_tilde)}"
        )

    print(accuracies)
