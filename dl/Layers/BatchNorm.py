import numpy as np

from dl.nn import Module
from dl.graph import Variable, batchNorm


class BatchNormLayer(Module):
    """
    BatchNorm Layer object.
    """
    def __init__(self, dim: int, eps=1e-5, momentum=0.1):
        """
        BatchNorm Layer object.

        Parameters
        ----------
        dim:
            input dimension.
        eps:
            epsilon to avoid divide by zero
        momentum:
            momentum used to compute moving average of mean and stddev.
        """
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

    def forward(self, x: Variable) -> Variable:
        """
        Process BatchNorm operations.

        Compute the moving average of mean and stddev, apply normalization and shifting.

        Parameters
        ----------
        x:
            Input

        Returns
        -------
        out:
            BN Result.
        """
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
        """
        Set the layer to evaluation mode. In this mode the moving average of mean and stddev will not be
        updated.

        Returns
        -------
        out:
            None
        """
        self._eval = True

    def train(self):
        """
        Set the layer to training mode. In this mode the moving average of mean and stddev will be
        updated.

        Returns
        -------
        out:
            None
        """
        self._eval = False


