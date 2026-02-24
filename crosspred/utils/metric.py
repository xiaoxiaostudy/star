import numpy as np


def rmse(test_data, predicted):
    """Calculate root mean squared error, ignoring missing values."""
    I = ~np.isnan(test_data)
    N = I.sum()
    sqerror = abs(test_data - predicted) ** 2
    mse = sqerror[I].sum() / N
    return np.sqrt(mse)
