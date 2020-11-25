import math

import numpy as np


def MAE(y_true, y_pre):
    y_true = (y_true).detach().numpy().copy().reshape((-1, 1))
    y_pre = (y_pre).detach().numpy().copy().reshape((-1, 1))
    re = np.abs(y_true - y_pre).mean()
    return re


def RMSE(y_true, y_pre):
    y_true = (y_true).detach().numpy().copy().reshape((-1, 1))
    y_pre = (y_pre).detach().numpy().copy().reshape((-1, 1))
    re = math.sqrt(((y_true - y_pre) ** 2).mean())
    return re


def MAPE(y_true, y_pre):
    y_true = (y_true).detach().numpy().copy().reshape((-1, 1))
    y_pre = (y_pre).detach().numpy().copy().reshape((-1, 1))
    e = (y_true + y_pre) / 2 + 1e-2
    re = (np.abs(y_true - y_pre) / (np.abs(y_true) + e)).mean()
    return re
