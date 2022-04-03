import numpy as np

from dl.nn import Module
from dl.graph import Variable, batchNorm


class BatchNormLayer(Module):

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.dim = dim
        self.gamma = Variable(np.ones((dim, 1)))
        self.beta = Variable(np.zeros((dim, 1)))
        self.epsilon = eps
        self.momentum = momentum
        self.variables = [self.gamma, self.beta]
        self.running_mean = None
        self.running_stdv = None
        self._eval = False

    def forward(self, x) -> Variable:
        if not self._eval:
            mean = np.mean(x.item, axis=1).reshape(self.dim, 1)
            var = np.var(x.item, axis=1).reshape(self.dim, 1)

            if self.running_mean is None:
                self.running_mean = mean
                self.running_stdv = var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_stdv = (1 - self.momentum) * self.running_stdv + self.momentum * var

        normalized = batchNorm(x, self.running_mean, self.running_stdv, self.epsilon)

        return normalized * self.gamma + self.beta

    def eval(self):
        self._eval = True

    def train(self):
        self._eval = False


