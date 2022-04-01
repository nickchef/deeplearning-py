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
