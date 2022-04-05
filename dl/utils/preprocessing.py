import numpy as np


def one_hot_encode(data, classes=None):
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
        self.x_max = None
        self.x_min = None

    def fit(self, data):
        x = np.vstack(data)
        self.x_max = np.max(x, axis=0)
        self.x_min = np.min(x, axis=0)
        ret = self.transform(data)
        return ret

    def transform(self, data):
        ret = data.copy()
        for i in range(len(ret)):
            ret[i] = (ret[i] - self.x_min) / (self.x_max - self.x_min)
        return ret


class StandardScaler:

    def __init__(self, eps=1e-8):
        self.x_mean = None
        self.x_stddev = None
        self.eps = eps

    def fit(self, data):
        x = np.vstack(data)
        self.x_mean = np.mean(x, axis=0)
        self.x_stddev = np.std(x, axis=0)
        return self.transform(data)

    def transform(self, data):
        ret = data.copy()
        for i in range(len(ret)):
            ret[i] = (ret[i] - self.x_mean) / (self.x_stddev + self.eps)
        return ret
