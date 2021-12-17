#  import numpy as np
import os
import matplotlib
import pandas as pd
import seaborn as sns

#  from matplotlib import cm
import matplotlib.pyplot as plt

#  from matplotlib.ticker import LinearLocator, FormatStrFormatter

sns.set()

import tikzplotlib


def tweak_tikz_plots(filename):
    """Tweaks the tikz plots to make them look better

    Parameters
    ----------
        filename : str
            The filename of the tikz plot to be tweaked
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    with open(filename, "w") as f:
        for line in lines:
            if "majorticks" in line:
                f.write(line.replace("false", "true"))
            elif "addplot" in line:
                f.write(line.replace("semithick", "thick"))
            elif "\\begin{axis}[" in line:
                f.write(line)
                f.write("width=12cm,")
                f.write("height=8cm,")
            else:
                f.write(line)


def save_tikz(filename, preview=False):
    """Saves the plot as a tikz-tex file

    Parameters
    ----------
        filename : str
            The filename of the tikz plot to be saved
    """
    plt.grid(True)
    tikzplotlib.clean_figure()
    tikzplotlib.save(filename)
    tweak_tikz_plots(filename)
    if preview:
        plt.show()
    #  tweak_tikz_plots(filename)
    plt.clf()


def line_plot(
    title,
    x_datas,
    y_datas,
    data_labels,
    x_label,
    y_label,
    x_log=False,
    y_log=False,
    filename="",
    show=True,
):
    """Plots a line plot

    Parameters
    ----------
        title : str
            The title of the plots
        x_datas : float[]
            The x data for the plot
        y_datas : float[]
            The y data for the plot
        data_labels : str[]
            The labels for the plot
        x_label : str
            The label for the x-axis
        y_label : str
            The label for the y-axis
        filename : str/None
            The filename for which to save the plot, does not save if None
    """
    plt.title(title)
    for x_data, y_data, label in zip(x_datas, y_datas, data_labels):
        sns.lineplot(x=x_data, y=y_data, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_log:
        plt.xscale("log")
    if y_log:
        plt.yscale("log")
    if filename:
        root, ext = os.path.splitext(filename)
        if ext == "":
            filename += ".tex"
        elif ext != ".tex":
            filename = root + ".tex"
        save_tikz(filename, show)
    elif show:
        plt.show()


def output_model_performance():
    """Outputs the model performance to a csv file"""
    filename, x_name = "test_size_performance.csv", "test_size"

    df = pd.read_csv(f"output/data/{filename}")
    x = df[x_name]
    y = df.loc[:, df.columns != x_name]

    features = ["accuracy", "TPR", "TNR", "PPV", "NPV", "FPR", "FNR", "FDR", "F1"]
    nice_features = ["Acc", "TPR", "TNR", "PPV", "NPV", "FPR", "FNR", "FDR", "F1"]
    model_names = ["logistic", "cnn with transform", "cnn without transform", "ffnn"]
    nice_model_names = [
        "Logistic",
        "CNN with augmentation",
        "CNN without augmentation",
        "FFNN",
    ]

    nice_table = f"""\\begin{{tabular}}{{ |p{{3cm}}||{'c|'*(len(features))}  }}
\\hline
\\multicolumn{{{len(features) + 1}}}{{|c|}}{{Performance of models}} \\\\
\\hline
 & {" & ".join(nice_features)} \\\\
\\hline\n"""

    for nice_model_name, model_name in zip(nice_model_names, model_names):
        nice_table += f"{nice_model_name} & "
        for feature in features:
            nice_table += f"{y[model_name + '_' +feature].mean()*100:6.2f}\\% & "
        nice_table = nice_table[:-2]
        nice_table += "\\\\\n\\hline \n"
    nice_table += "\\end{tabular}"
    #      nice_table += """\\hline
    #  \\end{tabular}"""

    print(nice_table)


def make_bias_variance_plots(show):
    """Makes the bias variance plots

    Parameters
    ----------
        show : bool
            Whether to show the plots
    """
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
            show=show,
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
            show=show,
        )


def main():
    output_model_performance()
    make_bias_variance_plots(False)
