from data import FrankeData
from regression import OrdinaryLeastSquares
from bias_variance import bootstrap_bias_variance
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib.pyplot as plt
import numpy as np


#  labels = ['G1', 'G2', 'G3', 'G4', 'G5']
#  men_means = [20, 34, 30, 35, 27]
#  women_means = [25, 32, 34, 20, 25]
#
#  x = np.arange(len(labels))  # the label locations
#  width = 0.35  # the width of the bars
#
#  fig, ax = plt.subplots()
#  rects1 = ax.bar(x - width/2, men_means, width, label='Men')
#  rects2 = ax.bar(x + width/2, women_means, width, label='Women')
#
#  # Add some text for labels, title and custom x-axis tick labels, etc.
#  ax.set_ylabel('Scores')
#  ax.set_title('Scores by group and gender')
#  ax.set_xticks(x, labels)
#  ax.legend()
#
#  ax.bar_label(rects1, padding=3)
#  ax.bar_label(rects2, padding=3)
#
#  fig.tight_layout()
#
#  plt.show()


def main():
    labels = []

    biases, variances = [], []
    degrees = list(range(1, 11))
    for degree in degrees:
        data = FrankeData(2000, degree=degree, test_size=0.2)
        bias, variance = bootstrap_bias_variance(OrdinaryLeastSquares(), data, 1000)
        labels.append(f"OLS degree: {degree}")
        biases.append(bias)
        variances.append(variance)
        print(f"OLS degree: {degree}")

    #  data = FrankeData(2000, test_size=0.2)
    #  reg = GradientBoostingRegressor(random_state=0)
    #  bias, variance = bootstrap_bias_variance(reg, data, 100)
    #  labels.append(f"Gradient boosting")
    #  biases.append(bias)
    #  variances.append(variance)
    #  degrees.append(11)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, biases, width, label="Bias")
    rects2 = ax.bar(x + width / 2, variances, width, label="Variance")

    ax.set_ylabel("Scores")
    ax.set_title("Scores by group and gender")
    #  ax.set_xticks(x, labels)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()
    print(labels)

    #  ax.bar_label(rects1, padding=3)
    #  ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

    #  import matplotlib.pyplot as plt
    #
    #  plt.plot(biases, label="bias")
    #  plt.plot(variances, label="variance")
    #  plt.legend()
    #  plt.show()
