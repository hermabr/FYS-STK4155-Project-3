import pandas as pd
import matplotlib.pyplot as plt

from plot import line_plot

#  filename, x_name = "bias_variance_ols.csv", "degree"
#  filename, x_name = "bias_variance_mlp_layer_size.csv", "layer size"
#  filename, x_name = "bias_variance_mlp_number_of_layers.csv", "number of layers"
#  filename, x_name = "bias_variance_ensamble.csv", "depth"

filename, x_name = "test_size_performance.csv", "test_size"

filenames = [
    "bias_variance_ols.csv",
    "bias_variance_mlp_layer_size.csv",
    "bias_variance_mlp_number_of_layers.csv",
    "bias_variance_ensamble.csv",
]
x_names = ["degree", "layer size", "number of layers", "depth"]


#  def line_plot(
#      title,
#      x_datas,
#      y_datas,
#      data_labels,
#      x_label,
#      y_label,
#      x_log=False,
#      y_log=False,
#      filename="",
#      show=True,
#  ):
#      pass


for filename, x_name in zip(filenames, x_names):
    #  df = pd.read_csv(filename)
    df = pd.read_csv(f"output/data/{filename}")
    y_values = df.loc[:, df.columns != x_name].values.transpose()
    x_values = [df[x_name].values for _ in range(len(y_values))]

    line_plot(
        f"{filename}",
        x_values,
        y_values,
        [f"{x_name} {i}" for i in range(len(y_values))],
        x_name,
        "bias variance",
        #  filename=f"output/plots/{filename}",
        show=True,
    )
    #  print(y_values)
    #  print([df[x_name].values for _ in range(len(y_values))])
    #  #  print()
    #  #  #  y = df.loc[:, df.columns != x_name].values
    #  exit()
    #  df.plot(x_name)
    #  plt.show()
