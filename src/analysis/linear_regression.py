from data import FallData
from regression import OrdinaryLeastSquares


def main():
    data = FallData(test_size=0.2, for_pytorch=False)

    ols = OrdinaryLeastSquares()
    ols.fit(data.X_train, data.y_train)
    z_tilde = ols.predict(data.X_test)
    print(data.y_test)
    print(z_tilde)
    print(f"{sum(z_tilde == data.y_test) / len(z_tilde) * 100:.0f}")
