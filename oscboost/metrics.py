"""Error metrics used in the thesis."""

import numpy as np


def wmape(y_true, y_pred):
    """Weighted mean absolute percentage error, sum|y - y_hat| / sum|y|."""
    y_true = np.asarray(y_true, dtype=float)
    return np.sum(np.abs(y_true - np.asarray(y_pred))) / np.sum(np.abs(y_true))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
