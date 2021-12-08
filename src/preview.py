import pandas as pd
import matplotlib.pyplot as plt

from plot import line_plot

#  filename, x_name = "bias_variance_ols.csv", "degree"
#  filename, x_name = "bias_variance_mlp_layer_size.csv", "layer size"
#  filename, x_name = "bias_variance_mlp_number_of_layers.csv", "number of layers"
#  filename, x_name = "bias_variance_ensamble.csv", "depth"

filename, x_name = "test_size_performance.csv", "test_size"

df = pd.read_csv(f"output/data/{filename}")
x = df[x_name]
y = df.loc[:, df.columns != x_name]

for feature in ["accuracy", "TPR", "TNR", "PPV", "NPV", "FPR", "FNR", "FDR", "F1"]:
    #  print(df.mean())
    for column in y.columns:
        if feature in column:
            print(column, df[column].mean())
    print()

file_informations = [
    {
        "filename": "bias_variance_ols.csv",
        "x_name": "degree",
        "pretty_name": "Bias Variance-tradeoff OLS",
    },
    {
        "filename": "bias_variance_mlp_layer_size.csv",
        "x_name": "layer size",
        "pretty_name": "Bias Variance-tradeoff MLP layer size",
    },
    {
        "filename": "bias_variance_mlp_number_of_layers.csv",
        "x_name": "number of layers",
        "pretty_name": "Bias Variance-tradeoff MLP hidden layers",
    },
    {
        "filename": "bias_variance_ensamble.csv",
        "x_name": "depth",
        "pretty_name": "Bias Variance-tradeoff Ensamble",
    },
]

for file_information in file_informations:
    df = pd.read_csv(f"output/data/{file_information['filename']}")

    x_values = [df[file_information["x_name"]].values for _ in range(2)]

    biases = df["bias"].values
    variances = df["variance"].values

    line_plot(
        f"{file_information['pretty_name']} Bias/Variance",
        x_values,
        [biases, variances],
        ["bias", "variance"],
        file_information["x_name"],
        "Error",
        filename=f"output/plots/{file_information['filename'][:-4]}_bias_variance",
        show=True,
    )

    # Plot train and test mse
    train_mse = df["mse_train"].values
    test_mse = df["mse_test"].values

    line_plot(
        f"{file_information['pretty_name']} MSE",
        x_values,
        [train_mse, test_mse],
        ["mse train", "mse test"],
        file_information["x_name"],
        "Error",
        filename=f"output/plots/{file_information['filename'][:-4]}_mse",
        show=True,
    )
