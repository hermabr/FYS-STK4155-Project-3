import config
import numpy as np
from tqdm import tqdm
from data import FallData
from analysis import pytorch
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def accuracy_metrics(y_true, y_pred):
    """Compute accuracy metrics for a given set of predictions and labels.

    Parameters
    ----------
        y_true : numpy.ndarray
            The true labels.
        y_pred : numpy.ndarray
            The predicted labels.

    Returns
    -------
        accuracy : float
            The accuracy of the model.
    """
    return sum(y_pred == y_true) / len(y_pred)


def get_all_metrics(y_true, y_pred):
    """Compute all metrics for a given set of predictions and labels.

    Parameters
    ----------
        y_true : numpy.ndarray
            The true labels.
        y_pred : numpy.ndarray
            The predicted labels.

    Returns
    -------
        accuracy : float
            The accuracy of the model.
        True Positive Rate : float
            The true positive rate of the model.
        True Negative Rate : float
            The true negative rate of the model.
        Positive Predictive Value : float
            The positive predictive value of the model.
        Negative Predictive Value : float
            The negative predictive value of the model.
        False Positive Rate : float
            The false positive rate of the model.
        False Negative Rate : float
            The false negative rate of the model.
        False Discovery Rate : float
            The false discovery rate of the model.
        F1 Score : float
            The F1 score of the model.
    """
    TN_FP, FN_TP = confusion_matrix(y_true, y_pred)
    TN, FP = TN_FP
    FN, TP = FN_TP

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    FDR = FP / (TP + FP)

    ACC = (TP + TN) / (TP + FP + FN + TN)

    F1 = 2 * PPV * TPR / (PPV + TPR)

    return f"{ACC},{TPR},{TNR},{PPV},{NPV},{FPR},{FNR},{FDR},{F1}"


def write_out(name, accuracy):
    """Write out the accuracy of the model to a file.

    Parameters
    ----------
        name : str
            The name of the model.
        accuracy : float
            The accuracy of the model.

    Returns
    -------
        The accuracy in a nice string format
    """
    return f"{float(accuracy)*100:6.2f}% "


def main():
    """Main function for the compare_models.py script."""
    accuracy_texts = []
    N = 1000

    for i in tqdm(range(N)):
        accuracy_text = f"{i},"
        outputs = f"{i:3}, "

        data = FallData(test_size=config.TEST_SIZE)
        data_pytorch_transform = FallData(
            test_size=config.TEST_SIZE,
            batch_size=config.BATCH_SIZE,
            for_pytorch=True,
            transform=True,
        )
        data_pytorch_no_transform = FallData(
            test_size=config.TEST_SIZE,
            batch_size=config.BATCH_SIZE,
            for_pytorch=True,
            transform=False,
        )

        # Logistic regression
        logreg = LogisticRegression()
        logreg.fit(data.X_train, data.y_train)
        z_tilde = logreg.predict(data.X_test)
        accuracy_text += f"{get_all_metrics(data.y_test, z_tilde)},"
        outputs += write_out("Log", get_all_metrics(data.y_test, z_tilde).split(",")[0])

        # Pytorch cnn with transform
        accuracy_pytorch_transform, losses, y_test, y_tilde = pytorch.train_model(
            data_pytorch_transform
        )
        accuracy_text += f"{get_all_metrics(y_test, y_tilde)},"
        outputs += write_out("CNN w", accuracy_pytorch_transform)

        # Pytorch cnn without transform
        accuracy_pytorch_no_transform, losses, y_test, y_tilde = pytorch.train_model(
            data_pytorch_no_transform
        )
        accuracy_text += f"{get_all_metrics(y_test, y_tilde)},"
        outputs += write_out("CNN n", accuracy_pytorch_no_transform)

        # Feed forward neural network
        net = MLPClassifier(
            solver="lbfgs",
            alpha=1e-3,
            hidden_layer_sizes=(100, 100, 100, 100),
            random_state=1,
        )
        net.fit(data.X_train, data.y_train)
        z_tilde = net.predict(data.X_test)
        accuracy_text += f"{get_all_metrics(data.y_test, z_tilde)}"
        outputs += write_out(
            "FFNN", get_all_metrics(data.y_test, z_tilde).split(",")[0]
        )
        tqdm.write(outputs)

        accuracy_texts.append(accuracy_text)

    # write the performance in the different metrics for the different models to a file
    with open("output/data/test_size_performance.csv", "w") as f:
        metric_text = "test_size,"
        for model_name in [
            "logistic",
            "cnn with transform",
            "cnn without transform",
            "ffnn",
        ]:
            for metric_name in [
                "accuracy",
                "TPR",
                "TNR",
                "PPV",
                "NPV",
                "FPR",
                "FNR",
                "FDR",
                "F1",
            ]:
                metric_text += f"{model_name}_{metric_name},"
        f.write(metric_text[:-1] + "\n")

        for line in accuracy_texts:
            f.write(line + "\n")
