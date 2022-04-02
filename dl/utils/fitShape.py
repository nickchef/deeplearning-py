import numpy as np


def fit_shape(arr, target_arr):
    if arr.shape != target_arr.shape:
        target_axis = 0 if arr.shape[0] != target_arr.shape[0] else 1
        return np.atleast_2d(np.sum(arr, axis=target_axis)).T
    else:
        return arr
