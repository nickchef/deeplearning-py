import numpy as np


def accuracy(y, yhat):
    yhat_max = np.max(yhat, axis=0)
    yres = np.where(yhat == yhat_max, 1, 0)
    right_cases = 0
    for i in range(y.shape[1]):
        right_cases += 1 if y[:, i].all() == yres[:i].all() else 0
