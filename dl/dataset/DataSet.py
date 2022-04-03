import numpy as np
from dl.graph.variable import Variable


class DataSet(object):

    def __init__(self, x, y, batch_size=1, shuffle=False, keep_remainder=False, T=True):
        self.x = []
        self.y = []
        self._idx = 0

        if shuffle:
            data_pair = np.shuffle(list(zip(x, y)))
            x = [i[0] for i in data_pair]
            y = [i[1] for i in data_pair]

        if batch_size > 1:
            def _make_batch(idx, bSize):
                x_b = np.array(x[idx])
                y_b = np.array(y[idx])
                for i in range(1, bSize):
                    x_b = np.vstack((x_b, np.array(x[idx + i])))
                    y_b = np.vstack((y_b, np.array(y[idx + i])))
                return x_b, y_b

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
            self.x = [Variable(np.array(i).T if T else np.array(i), no_grad=True) for i in x]
            self.y = [Variable(np.array(i).T if T else np.array(i), no_grad=True) for i in y]

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