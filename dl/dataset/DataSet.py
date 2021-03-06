from typing import Iterable

import numpy as np
from dl.graph.variable import Variable


class DataSet(object):

    def __init__(self, x: Iterable, y: Iterable, batch_size=1, shuffle=False, keep_remainder=False, T=True) -> None:
        """

        A container of the data to achieve mini-batch and iterator.

        Parameters
        ----------
        x: Iterable
            Features of samples
        y: Iterable
            Labels of samples
        batch_size: int
            Mini-batch size
        shuffle: bool
            Set "True" to randomly reorder given data.
        keep_remainder: bool
            If the given samples size is not divisible by the batch size, set "True" to keep the remainder samples
        T: bool
            Set "True" to transpose the feature data.
        """
        self.x = []
        self.y = []
        self._idx = 0

        if shuffle:  # Shuffle the data, keep alignment between x and y
            data_pair = list(zip(x, y))
            np.random.shuffle(data_pair)
            x = [i[0] for i in data_pair]
            y = [i[1] for i in data_pair]

        def _make_batch(idx, bSize):  # make a single batch
            x_b = np.array(x[idx])
            y_b = np.array(y[idx])
            for i in range(1, bSize):
                x_b = np.vstack((x_b, np.array(x[idx + i])))
                y_b = np.vstack((y_b, np.array(y[idx + i])))
            return x_b, y_b

        if batch_size > 1:  # if mini-batch enabled

            available_size = len(x) - len(x) % batch_size
            for i in range(0, available_size, batch_size):
                x_batch, y_batch = _make_batch(i, batch_size)
                self.x.append(Variable(x_batch.T if T else x_batch, no_grad=True))
                self.y.append(Variable(y_batch.T if T else y_batch, no_grad=True))

            if keep_remainder and available_size != len(x):
                x_batch, y_batch = _make_batch(available_size, len(x) - available_size)
                self.x.append(Variable(x_batch.T if T else x_batch, no_grad=True))
                self.y.append(Variable(y_batch.T if T else y_batch, no_grad=True))
        else:
            x_batch, y_batch = _make_batch(0, len(x))
            self.x = [Variable(x_batch.T if T else x_batch, no_grad=True) for _ in x]
            self.y = [Variable(y_batch.T if T else y_batch, no_grad=True) for _ in y]

    def __iter__(self):
        self._idx = 0
        return self

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __next__(self):
        if self._idx < len(self.x):
            self._idx += 1
            return self.x[self._idx - 1], self.y[self._idx - 1]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.x)
