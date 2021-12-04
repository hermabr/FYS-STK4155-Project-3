import argparse
import numpy as np
from data import DigitsData, FrankeData

#  import neural_network
#  from ann import ANNModel
import cnn
from FFNN import FeedForwardNeuralNetwork
from layers import ReluLayer, LinearLayer

from bias_variance import bootstrap_bias_variance

from regression import OrdinaryLeastSquares, Ridge, Lasso

from analysis import bias_variance


def test_data():
    data = FrankeData(400, test_size=0.2)
    #  data = DigitsData(test_size=0.2)
    #  print(data.n_features)


if __name__ == "__main__":
    #  test_data()
    #  exit()

    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="To run the bias-variance tradeoff experiment"
    )
    parser.add_argument(
        "-bv",
        "--biasvariance",
        help="To test the bias-variance tradeoff for three different methods",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--all",
        help="To run all the analyzes",
        action="store_true",
    )

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

    if args.biasvariance or args.all:
        bias_variance.main()
        #  biases, variances = [], []
        #  degrees = list(range(1, 11))
        #  #  for degree in degrees:
        #  #      data = FrankeData(2000, degree=degree, test_size=0.2)
        #  #      bias, variance = bootstrap_bias_variance(
        #  #          OrdinaryLeastSquares(), data, 10_000
        #  #      )
        #  #      biases.append(bias)
        #  #      variances.append(variance)
        #  #      print(degree)
        #
        #  data = FrankeData(2000, test_size=0.2)
        #  net = FeedForwardNeuralNetwork(
        #      data.X_train.shape[1],
        #      [150] * 3,
        #      hidden_layers=ReluLayer,
        #      final_layer=LinearLayer,
        #      classification=False,
        #      epochs=1000,
        #      learning_rate=0.001,
        #      lambda_=0.05,
        #      verbose=True,
        #  )
        #
        #  bias, variance = bootstrap_bias_variance(net, data, 10_000)
        #  biases.append(bias)
        #  variances.append(variance)
        #
        #  import matplotlib.pyplot as plt
        #
        #  plt.plot(degrees, biases, label="Bias")
        #  plt.plot(degrees, variances, label="Variance")
        #  plt.legend()
        #  plt.show()
        #  #  plt.bar(degrees, biases, label="Bias")
        #  #  plt.legend()
        #  #  plt.show()
        #  #  plt.plot(degrees, variances, label="Variance")
        #  #  plt.legend()
        #  #  plt.show()

    #  parser.add_argument(
    #      "-d",
    #      "--data",
    #      help="To Test the data loading",
    #      action="store_true",
    #  )
    #  parser.add_argument(
    #      "-l",
    #      "--linear",
    #      help="To run the analysis for the linear regression based model",
    #      action="store_true",
    #  )
    #  parser.add_argument(
    #      "-n",
    #      "--neuralnet",
    #      help="To run the analysis for the neural network based model",
    #      action="store_true",
    #  )
    #  parser.add_argument(
    #      "-t",
    #      "--tree",
    #      help="To run the analysis for the tree-based model",
    #      action="store_true",
    #  )
