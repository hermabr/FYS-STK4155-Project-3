import numpy as np
from tqdm import tqdm
from data import FallData
from analysis import pytorch

#  from cnn.tensorflow import get_model
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def accuracy_metrics(y_true, y_pred):
    return sum(y_pred == y_true) / len(y_pred)


def get_all_metrics(y_true, y_pred):
    #  tn_fp, fn_tp = confusion_matrix(true_output, prediction_log_reg)
    #  tn, fp = tn_fp
    #  fn, tp = fn_tp
    #
    #  tn, fp = tn_fp
    #  fn, tp = fn_tp
    #
    #  ppv = tp / (tp + fp) * 100
    #  npv = tn / (tn + fn) * 100
    #
    #  F1_score = tp / (tp + 0.5 * (fp + fn))
    #
    #  print(f"PPV: {ppv:.2f}%, NPV: {npv:.2f}%, F1: {F1_score:.2f}, Accuracy: {accuracy_metrics(true_output, prediction_log_reg):.2f}")

    TN_FP, FN_TP = confusion_matrix(y_true, y_pred)
    TN, FP = TN_FP
    FN, TP = FN_TP
    #  FP = CM.sum(axis=0) - np.diag(CM)
    #  FN = CM.sum(axis=1) - np.diag(CM)
    #  TP = np.diag(CM)
    #  TN = CM.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    # F1 score
    F1 = 2 * PPV * TPR / (PPV + TPR)

    return f"{ACC},{TPR},{TNR},{PPV},{NPV},{FPR},{FNR},{FDR},{F1}"


def write_out(name, accuracy):
    #  tqdm.write(f"{name}:\t Accuracy: {float(accuracy)*100:.2f}%")
    #  tqdm.write(f"{float(accuracy)*100:.2f}% ", end=" ")
    return f"{float(accuracy)*100:6.2f}% "


def main():
    accuracy_texts = []
    N = 1000

    for i in tqdm(range(N)):
        accuracy_text = f"{i},"
        outputs = f"{i:3}, "

        #  for i in range(1):
        test_size = 0.2
        #  for test_size in tqdm(np.arange(0.1, 1, 0.1)):
        data = FallData(test_size=test_size)
        data_pytorch_transform = FallData(
            test_size=test_size, batch_size=4, for_pytorch=True, transform=True
        )
        data_pytorch_no_transform = FallData(
            test_size=test_size, batch_size=4, for_pytorch=True, transform=False
        )

        #  data_tensorflow = FallData(test_size=test_size, for_tensorflow=True)

        #  # Logistic regression
        logreg = LogisticRegression()
        logreg.fit(data.X_train, data.y_train)
        z_tilde = logreg.predict(data.X_test)
        accuracy_text += f"{get_all_metrics(data.y_test, z_tilde)},"
        #  accuracy_logistic = accuracy_metric(data.y_test, z_tilde)
        outputs += write_out("Log", get_all_metrics(data.y_test, z_tilde).split(",")[0])

        #  # Pytorch cnn with transform
        accuracy_pytorch_transform, losses, y_test, y_tilde = pytorch.train_model(
            data_pytorch_transform
        )
        accuracy_text += f"{get_all_metrics(y_test, y_tilde)},"
        outputs += write_out("CNN w", accuracy_pytorch_transform)

        #  # Pytorch cnn without transform
        accuracy_pytorch_no_transform, losses, y_test, y_tilde = pytorch.train_model(
            data_pytorch_no_transform
        )
        accuracy_text += f"{get_all_metrics(y_test, y_tilde)},"
        outputs += write_out("CNN n", accuracy_pytorch_no_transform)

        # Tensorflow cnn
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
        #  write_out("Tensorflow", accuracy_tensorflow)

        # Feed forward neural network
        accuracy_ffnn = 0.2
        net = MLPClassifier(
            solver="lbfgs",
            alpha=1e-3,
            hidden_layer_sizes=(100, 100, 100, 100),
            random_state=1,
        )
        net.fit(data.X_train, data.y_train)
        z_tilde = net.predict(data.X_test)
        accuracy_text += f"{get_all_metrics(data.y_test, z_tilde)}"
        #  accuracy_ffnn = accuracy_metric(data.y_test, z_tilde)
        outputs += write_out(
            "FFNN", get_all_metrics(data.y_test, z_tilde).split(",")[0]
        )
        tqdm.write(outputs)

        #  if (
        #      accuracy_logistic == accuracy_pytorch
        #      and accuracy_logistic == accuracy_ffnn
        #      and accuracy_logistic == 1
        #  ):
        #      print("HA MA GAD")
        #      exit()

        #  accuracy_texts.append(
        #      f"{i},{accuracy_logistic*100},{accuracy_pytorch_transform*100},{accuracy_pytorch_no_transform*100},{accuracy_ffnn*100}\n"
        #      #  f"{i}){test_size:.1f},{accuracy_logistic*100},{accuracy_pytorch*100},{accuracy_ffnn*100}\n"
        #  )

        #  f.write(
        #      f"{test_size*100:.0f},{accuracy_logistic*100:.2f},{accuracy_pytorch*100:.2f},{accuracy_tensorflow*100:.2f},{accuracy_ffnn*100:.2f}\n"
        #  )
        accuracy_texts.append(accuracy_text)

    with open("output/data/test_size_performance.csv", "w") as f:
        #  f.write("test_size,logistic,cnn with transform,cnn without transform,ffnn\n")
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
                #  f.write(f"{model_name}_{metric_name}")
        f.write(metric_text[:-1] + "\n")

        #  f.write("test_size,logistic,pytorch cnn,ffnn\n")
        for line in accuracy_texts:
            f.write(line + "\n")
