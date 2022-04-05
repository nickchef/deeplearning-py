import numpy as np
from dl.graph.variable import Variable


def accuracy(y: Variable, yhat: Variable) -> float:
    """
    Compute the classification accuracy.

    Parameters
    ----------
    y:
        Prediction.
    yhat:
        Ground Truth.

    Returns
    -------
    out:
        The rate of correct prediction.
    """
    yhat_max = np.max(yhat.item, axis=0)  # Process the predicted class.
    yres = np.where(yhat.item == yhat_max, 1, 0)
    right_cases = 0
    for i in range(y.shape[1]):
        right_cases += 1 if np.all(y.item[:, i] == yres[:, i]) else 0
    return right_cases / y.shape[1]
