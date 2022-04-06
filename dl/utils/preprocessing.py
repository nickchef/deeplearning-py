import numpy as np


def one_hot_encode(data, classes=None):
    """
    One hot encoding. For all the category exist in data, enlarge all the data to the dimension of category number,
    and set the index of their own category to 1.

    Example:
    one_hot_encode([[1], [2], [3]]) = [[1,0,0], [0,1,0], [0,0,1]]

    Parameters
    ----------
    data:
        Raw data to be encoded.
    classes:
        Class number in the raw data. Set to None to automatically detect. Cannot be lower than actual exist classes.
    Returns
    -------
    Encoded:
        Encoded data.
    """
    class_names = np.unique(data)
    classes_found = class_names.shape[0]
    if classes is None:
        classes = classes_found
    elif classes_found > classes:
        raise ValueError(f"Classes found {classes_found} greater than {classes}")

    class_dict = {}
    for idx, name in enumerate(class_names):
        class_dict[name] = idx

    encoded = [[0 for _ in range(classes)] for _ in data]
    for idx, val in enumerate(data):
        encoded[idx][class_dict[int(val)]] = 1

    return encoded


class Normalizer:

    def __init__(self):
        """
        Min-max normalization. Normalized the data to [0-1] and keep the distribution of raw data.

        x = (x - xmin)/(xmax - xmin)
        """
        self.x_max = None
        self.x_min = None

    def fit(self, data):
        """
        Normalize the data and record the max/min value.

        Parameters
        ----------
        data:
            Data to be normalized.

        Returns
        -------
        data:
            Normalized data.
        """
        x = np.vstack(data)
        self.x_max = np.max(x, axis=0)
        self.x_min = np.min(x, axis=0)
        ret = self.transform(data)
        return ret

    def transform(self, data):
        """
        Normalize the data based on .

        Parameters
        ----------
        data:
            Data to be normalized.

        Returns
        -------
        data:
            Normalized data.
        """
        if self.x_min is None or self.x_max is None:
            raise RuntimeError("Scaler has not been fitted.")
        ret = data.copy()
        for i in range(len(ret)):
            ret[i] = (ret[i] - self.x_min) / (self.x_max - self.x_min)
        return ret


class StandardScaler:

    def __init__(self, eps=1e-8):
        """
        Standardization. Transform the data to 0 mean and 1 stddev.

        x = (x - mean(x))/std(x)

        Parameters
        ----------
        eps:
            A very small number to avoid divide by zero.
        """
        self.x_mean = None
        self.x_stddev = None
        self.eps = eps

    def fit(self, data):
        """
        Standardize the data and record its mean and stddev.

        Parameters
        ----------
        data:
            Data to be standardized.

        Returns
        -------
        data:
            Standardized Data.
        """
        x = np.vstack(data)
        self.x_mean = np.mean(x, axis=0)
        self.x_stddev = np.std(x, axis=0)
        return self.transform(data)

    def transform(self, data):
        """
        Standardize the data.

        Parameters
        ----------
        data:
            Data to be standardized.

        Returns
        -------
        data:
            Standardized Data.
        """
        ret = data.copy()
        for i in range(len(ret)):
            ret[i] = (ret[i] - self.x_mean) / (self.x_stddev + self.eps)
        return ret
