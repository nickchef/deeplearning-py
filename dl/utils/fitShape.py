import numpy as np


def fit_shape(arr, target_arr) -> np.ndarray:
    """
    Fit the shape of a matrix with the other matrix.

    Parameters
    ----------
    arr:
        Matrix need to be fit.
    target_arr
        Target matrix to be fit with.
    Returns
    -------
    mat:
        Fitted matrix.
    """
    if arr.shape != target_arr.shape:
        target_axis = 0 if arr.shape[0] != target_arr.shape[0] else 1
        return np.atleast_2d(np.sum(arr, axis=target_axis)).T
    else:
        return arr
