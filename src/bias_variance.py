import numpy as np


def bootstrap_bias_variance(model, data, bootstrap_N):
    """Bootstrap method

    Parameters
    ----------
        data : data_generation.Data
            The data for which to do the bootstrap
        bootstrap_N : int
            The number of bootstrap samples to run

    Returns
    -------
        bootstrap_z_tilde: np.array
            An array containing z_tildes generated from the bootstrap
    """
    N = len(data.X_train)

    #  bootstrap_z_tilde = np.zeros((bootstrap_N, N))
    bootstrap_z_tilde = np.empty((bootstrap_N, len(data.y_test)))

    for i in range(bootstrap_N):
        indices = np.random.randint(0, N, N)

        X_train = data.X_train[indices]
        y_train = data.y_train[indices]

        model.fit(X_train, y_train)
        bootstrap_z_tilde[i] = model.predict(data.X_test)

    return bias(data.y_test, bootstrap_z_tilde), variance(bootstrap_z_tilde)


def bias(z, z_tilde):
    """Calculates the bias

    Parameters
    ----------
        z : np.array
            The z-values for which to calculate the bias
        z_tilde : np.array
            The predicted z-tilde-values for which to calculate the bias

    Returns
    -------
        bias : float
            The bias of the z and z_tilde
    """
    return np.mean((z - np.mean(z_tilde, axis=0)) ** 2)


def variance(z_tilde):
    """Calculates the variance

    Parameters
    ----------
        z_tilde : np.array
            The predicted z-tilde-values for which to calculate the variance

    Returns
    -------
        variance : float
            The variance of the z_tilde
    """
    return np.mean(np.var(z_tilde, axis=1))
