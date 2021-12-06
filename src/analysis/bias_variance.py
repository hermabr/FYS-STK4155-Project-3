import numpy as np
from tqdm import tqdm
from data import FrankeData
import matplotlib.pyplot as plt
from regression import OrdinaryLeastSquares
from bias_variance import bootstrap_bias_variance
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor


def plot(mses, biases, variances, title=""):
    plt.plot(mses, label="mse")
    plt.plot(biases, label="bias")
    plt.plot(variances, label="variance")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def bias_variance_analysis_ols():
    mses, biases, variances = [], [], []
    degrees = list(range(1, 11))
    for degree in degrees:
        data = FrankeData(2000, degree=degree, test_size=0.2)
        bias, variance, mse = bootstrap_bias_variance(
            OrdinaryLeastSquares(), data, 1000
        )
        mses.append(mse)
        biases.append(bias)
        variances.append(variance)
        #  print(bias, variance, mse)

    # write mse bias and variance to file
    with open("output/data/bias_variance_ols.csv", "w") as f:
        f.write("degree,mse,bias,variance\n")
        for i in range(len(degrees)):
            f.write(f"{degrees[i]},{mses[i]},{biases[i]},{variances[i]}\n")
    #  plot(mses, biases, variances)


def bias_variance_analysis_mlp():
    mses, biases, variances = [], [], []
    data = FrankeData(2000, test_size=0.2)
    for size in range(10, 100, 10):
        bias, variance, mse = bootstrap_bias_variance(
            MLPRegressor(hidden_layer_sizes=(size for _ in range(3)), max_iter=1000),
            data,
            100,
        )
        mses.append(mse)
        biases.append(bias)
        variances.append(variance)

    plot(mses, biases, variances)


def bias_variance_analysis_ensamble():
    mses, biases, variances = [], [], []
    data = FrankeData(2000, test_size=0.2)
    #  for size in range(10, 1100, 100):
    for depth in range(1, 7):
        bias, variance, mse = bootstrap_bias_variance(
            GradientBoostingRegressor(max_depth=depth),
            data,
            100,
        )
        mses.append(mse)
        biases.append(bias)
        variances.append(variance)

        tqdm.write(f"{mse} {bias} {variance}")

    plot(mses, biases, variances, "A")


def main():
    bias_variance_analysis_ols()
    #  bias_variance_analysis_mlp()
    #  bias_variance_analysis_ensamble()
