from typing import Iterable

from dl.graph.variable import Variable
from dl.utils.fitShape import fit_shape
import numpy as np
from scipy.special import erf


class Operator(object):
    """
    Basic class for all the operator objects.
    """

    def __call__(self, *var) -> Variable:
        """

        Perform the computation.

        Parameters
        ----------
        var: Variable..
            Variables to perform the computation.

        Returns
        -------
        out: Variable
            An Variable object of computation result.

        """
        return self.compute(*var)

    def compute(self, *var) -> Variable:
        """

        Perform the computation.

        Parameters
        ----------
        var: Variable..
            Operands of the computation.

        Returns
        -------
        out: Variable
            An Variable object of computation result.

        """
        raise NotImplementedError

    def gradient(self, input_variables: list, prev_grad: np.ndarray) -> Iterable:
        """

        Calculate the gradient through chain rules.

        Parameters
        ----------
        input_variables: Variable..
            Operands of this computation.
        prev_grad
            Gradient from previous computation.

        Returns
        -------
        out: Iterable
            The gradient of all operands in this computation.
        """
        raise NotImplementedError


class Add(Operator):
    """
    Operator object of add.
    """

    def compute(self, *var) -> Variable:
        """

        Perform the add computation.

        Parameters
        ----------
        var: Variable..
            Variables to be added.

        Returns
        -------
        out: Variable
            An result Variable object.

        """
        return Variable(var[0].item + var[1].item,
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables: list, prev_grad: np.ndarray) -> Iterable:
        """

        Calculate the gradient through chain rules.
        For add operation, the gradient should be 1 for each operand.

        z = x + y , w = f(z), then

        dx/dw = dz/dw

        Parameters
        ----------
        input_variables
        prev_grad

        Returns
        -------

        """
        return fit_shape(prev_grad, input_variables[0]), fit_shape(prev_grad, input_variables[1])


class Mul(Operator):

    def compute(self, *var):
        return Variable(var[0].item * var[1].item,
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return fit_shape(input_variables[1].item * prev_grad, input_variables[0].item), \
               fit_shape(input_variables[0].item * prev_grad, input_variables[1].item)


class MatMul(Operator):
    # Input features should be transposed
    def compute(self, *var):
        return Variable(np.matmul(var[0].item, var[1].item),
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return np.matmul(prev_grad, input_variables[1].item.T), \
               np.matmul(input_variables[0].item.T, prev_grad)


class ReLU(Operator):
    def compute(self, *var):
        return Variable(np.where(var[0].item > 0, var[0].item, 0),
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        grad = np.where(input_variables[0].item > 0, 1, 0) * prev_grad
        return [grad]


class SoftMax(Operator):
    # For mini-batch, the output layout will be same with the input
    def compute(self, *var):
        return Variable(np.exp(var[0].item) / np.sum(np.exp(var[0].item), axis=0),
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return [prev_grad]  # No grad needed for output


class CrossEntropy(Operator):
    # y: var[0]
    # yhat: var[1]
    def compute(self, *var):
        return Variable(
            np.atleast_2d(-np.sum(var[0].item * np.log(var[1].item), axis=0) / var[0].item.shape[0]),
            input_vars=var,
            operator=self,
            no_grad=True
        )

    def gradient(self, input_variables, prev_grad):
        return 0, input_variables[1].item - input_variables[0].item


class Sub(Operator):

    def compute(self, *var):
        return Variable(var[0].item - var[1].item,
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return prev_grad, -prev_grad


class Dropout(Operator):

    def __init__(self, rate):
        self.rate = rate
        self.mask = None
        self.eval = False

    def maskGen(self, shape):
        self.mask = np.ones(shape)
        if not self.eval:
            input_nuerons = shape[0]
            dropout = int(input_nuerons * self.rate)
            choice = np.random.choice(input_nuerons, size=dropout, replace=False)
            for i in choice:
                self.mask[i] = np.zeros_like(self.mask[i])
            self.mask *= 1 / (1 - self.rate)

    def compute(self, *var):
        self.maskGen(var[0].shape)
        return Variable(var[0].item * self.mask, input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return [prev_grad * self.mask]


class BatchNorm(Operator):

    def compute(self, *var):  # val, mean, var, eps
        return Variable((var[0].item - var[1]) / np.sqrt(var[2] + var[3]),
                        input_vars=[var[0], Variable(np.sqrt(var[2] + var[3]), no_grad=True)],
                        operator=self,
                        no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return [prev_grad / input_variables[1].item]


class GELU(Operator):

    def compute(self, *var):
        x = var[0].item
        return Variable(0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))),
                        input_vars=var,
                        operator=self,
                        no_grad=True)

    def gradient(self, input_variables, prev_grad):
        x = input_variables[0].item
        return (0.5 * np.tanh(0.0356774 * np.power(x, 3) + 0.797885 * x) +
                (0.0535161 * np.power(x, 3) + 0.398942 * x) *
                np.power((1 / np.cosh(0.0356774 * np.power(x, 3) + 0.797885 * x)), 2) + 0.5) * prev_grad


def _p(var):
    return .5 * (1. + erf(var / np.sqrt(2.)))


class _GELU(Operator):

    def compute(self, *var):
        return Variable(var[0].item * _p(var[0].item),
                        input_vars=var,
                        operator=self,
                        no_grad=True)

    def gradient(self, input_variables, prev_grad):
        x = input_variables[0].item
        return (_p(x) + x / np.sqrt(np.pi) * np.exp(-np.power(x, 2) / 2)) * prev_grad


def relu(var):
    return ReLU()(var)


def softmax(var):
    return SoftMax()(var)


def batchNorm(x, mean, var, eps):
    return BatchNorm()(x, mean, var, eps)


def gelu(var, estimate=True):
    return GELU()(var) if estimate else _GELU()(var)


def do_nothing(var):
    return var
