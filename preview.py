import pandas as pd
import matplotlib.pyplot as plt

filename, x_name = "bias_variance_ols.csv", "degree"

#  df = pd.read_csv("./test_size_performance.csv")
#  df = pd.read_csv("output/data/bias_variance_ols.csv")
df = pd.read_csv(f"output/data/{filename}")


x = df[x_name].values
y = df.loc[:, df.columns != x_name].values
print(df)

df.plot(x_name)
plt.title(filename)
plt.show()
