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
        prev_grad: np.ndarray
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
        input_variables: Variable..
            Operands of this computation.
        prev_grad: np.ndarray
            Gradient from previous computation.

        Returns
        -------
        out: Iterable
            An tuple of gradients of two operands.

        """
        #  Gradient is reshaped to fit the operand.
        return fit_shape(prev_grad, input_variables[0]), fit_shape(prev_grad, input_variables[1])


class Mul(Operator):
    """
    Operator object of multiplication.
    """

    def compute(self, *var):
        """

        Perform the multiply computation.

        Parameters
        ----------
        var: Variable..
            Variables to be multiplied.

        Returns
        -------
        out: Variable
            An result Variable object.
        """
        return Variable(var[0].item * var[1].item,
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        """

        Calculate the gradient through chain rules.
        For multiply operation, the gradient should be the value of other operand for each operand.

        z = x * y , w = f(z), then

        dx/dw = y * dz/dw

        Parameters
        ----------
        input_variables: Variable..
            Operands of this computation.
        prev_grad: np.ndarray
            Gradient from previous computation.

        Returns
        -------
        out: Iterable
            An tuple of gradients of two operands.
        -------

        """
        return fit_shape(input_variables[1].item * prev_grad, input_variables[0].item),\
            fit_shape(input_variables[0].item * prev_grad, input_variables[1].item)


class MatMul(Operator):
    """
    Operator object of matrix multiplication.
    """
    def compute(self, *var):
        """

        Perform the matrix multiplication.

        Parameters
        ----------
        var: Variable..
            Operands of matrix multiplication.

        Returns
        -------
        out: Variable
            Result variable object.
        """
        return Variable(np.matmul(var[0].item, var[1].item),
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        """

        Calculate the gradient through chain rules.
        For matrix multiplication, the gradient should be the other operand dot multiply with previous gradient.

        z = x @ y , w = f(z), then

        dx/dw = dz/dw @ yT
        dy/dw = xT @ dz/dw

        Parameters
        ----------
        input_variables: Variable..
            Operands of this computation.
        prev_grad: np.ndarray
            Gradient from previous computation.

        Returns
        -------
        out: Iterable
            An tuple of gradients of two operands.
        """
        return np.matmul(prev_grad, input_variables[1].item.T), \
            np.matmul(input_variables[0].item.T, prev_grad)


class ReLU(Operator):
    """
    Operator object of ReLU operation.
    """
    def compute(self, *var) -> Variable:
        """

        Perform the ReLU operation.

        Parameters
        ----------
        var: Variable..
            The operand.

        Returns
        -------
        out: Variable
            Result Variable object.
        """
        return Variable(np.where(var[0].item > 0, var[0].item, 0),
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        """

        Calculate the gradient through chain rules.
        For relu, the gradient should 1 if the operand is greater than 0 else 0.

        z = relu(x) , w = f(z), then

        dx/dw = dz/dw if x > 0 else 0

        Parameters
        ----------
        input_variables: Variable..
            Operands of this computation.
        prev_grad: np.ndarray
            Gradient from previous computation.

        Returns
        -------
        out: Iterable
            An tuple of gradients of two operands.
        """
        grad = np.where(input_variables[0].item > 0, 1, 0) * prev_grad
        return [grad]


class SoftMax(Operator):
    """
    Operator object of SoftMax operation.
    """
    def compute(self, *var):
        """

        Perform the softmax operation.

        Parameters
        ----------
        var: Variable..
            The vector or matrix.

        Returns
        -------
        out: Variable
            A Variable object of Softmax result.
        """
        max_val = np.max(var[0].item, axis=0)
        """
        avoid underflow/overflow of exp operation.
        e^x1 / (e^x1 + ... + e^xi) = (e^(x1 - xmax)) / (e^(x1-xmax) +...+ e^(xi-xmax))
        """
        return Variable(np.exp(var[0].item-max_val) / np.sum(np.exp(var[0].item-max_val), axis=0),
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        """

        Calculate the gradient through chain rules.
        The gradient of softmax operation has be omitted and integrated into the loss function.

        Parameters
        ----------
        input_variables: Variable..
            Operands of this computation.
        prev_grad: np.ndarray
            Gradient from previous computation.

        Returns
        -------
        out: Iterable
            The graients of given vector or matrix.
        """
        return [prev_grad]  # No grad needed for output


class CrossEntropy(Operator):
    """
    Operator object of Cross Entropy operation.
    """
    def __init__(self, eps=1e-8):
        self.eps = eps

    def compute(self, *var):  # var = [y, yhat]
        """

        Perform the CrossEntropy computation.

        Parameters
        ----------
        var: Variable..
            [y, yhat]
        Returns
        -------
        out: Variable
            Cross Entropy loss.
        """

        softmax_input = var[1].input_vars[0].item
        input_max = np.max(softmax_input, axis=0)
        reduced_input = softmax_input - input_max
        exp_res = np.exp(reduced_input)
        exp_sum_other = np.sum(exp_res, axis=0)
        log_res_of_softmax = reduced_input - np.log(exp_sum_other)
        """
        Avoid log(0) error. 
        Log(e^x1 / (e^x1 + ... + e^xi)) = log((e^(x1 - xmax)) / (e^(x1-xmax) +...+ e^(xi-xmax))) = 
        log(e^(x1-xmax) - Log((e^(x1-xmax) +...+ e^(xi-xmax)))) = x1-xmax-Log((e^(x1-xmax) +...+ e^(xi-xmax))))
        """
        return Variable(
            np.mean(-np.sum(var[0].item * log_res_of_softmax, axis=0) / var[0].item.shape[0]),
            input_vars=var,
            operator=self,
            no_grad=True
        )

    def gradient(self, input_variables, prev_grad):
        """

        Calculate the gradient through chain rules.
        The gradient of Cross Entropy has been combined with the softmax.
        y = softmax(x), z = CE(z, ground truth)
        dx/dz = z - ground truth

        Parameters
        ----------
        input_variables: Variable..
            Operands of this computation.
        prev_grad: np.ndarray
            Gradient from previous computation.

        Returns
        -------
        out: Iterable
            The graients of the variables before softmax.
        """
        return 0, input_variables[1].item - input_variables[0].item  # No gradients needed for ground truth


class Sub(Operator):
    """
    Operator object of substraction. This operator has not been involved in current functions, but can be used in
    generall Variable caculations.
    """
    def compute(self, *var):
        """

        Perform the substraction.

        Parameters
        ----------
        var: Variable..
            Two operands of substraction.
        Returns
        -------
        out: Variable
            A Variable object of the difference of two operands.
        """
        return Variable(var[0].item - var[1].item,
                        input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        """

        Calculate the gradient through chain rules.
        The gradient of substraction is 1 and -1 for the two operands correspondly.
        z = x - y, w = f(z)
        dx/dw = dz/dw
        dy/dw = -dz/dw

        Parameters
        ----------
        input_variables: Variable..
            Operands of this computation.
        prev_grad: np.ndarray
            Gradient from previous computation.

        Returns
        -------
        out: Iterable
            The graients of the operands.
        """
        #  Gradient is reshaped to fit the operand.
        return fit_shape(prev_grad, input_variables[0]), fit_shape(-prev_grad, input_variables[1])


class Dropout(Operator):
    """
    Operator object of Dropout operation.
    """
    def __init__(self, rate: float) -> None:
        """

        Dropout object. Recording the mask used to perform the dropout operation.

        Parameters
        ----------
        rate: float
            Dropout rate.
        """
        self.rate = rate
        self.mask = None
        self.eval = False

    def maskGen(self, shape: tuple) -> None:
        """

        Generate mask for dropout.

        Parameters
        ----------
        shape: tuple

        Returns
        -------
        out:
            None
        """
        self.mask = np.ones(shape)  # if the model is being test, dropout will not be performed.
        if not self.eval:
            input_nuerons = shape[0]
            dropout = int(input_nuerons * self.rate)
            choice = np.random.choice(input_nuerons, size=dropout, replace=False)
            for i in choice:
                self.mask[i] = np.zeros_like(self.mask[i])
            self.mask *= 1 / (1 - self.rate)  # Enlarge the variables which has not been dropped.

    def compute(self, *var):
        """
        Perform Dropout operation.

        Parameters
        ----------
        var:
            Variable to be dropout-ed.

        Returns
        -------
        out:
            Variable has been dropout-ed
        """
        self.maskGen(var[0].shape)  # generate masks.
        return Variable(var[0].item * self.mask, input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        """
        Calculate the gradient of variable has been dropout-ed.
        The gradient of input should be equal to the mask used for previous dropout operation.

        y = dropout(x, mask), z = f(y)

        dx/dz = dy/dz * mask

        Parameters
        ----------
        input_variables:
            Input of the dropout operation.
        prev_grad:
            Gradient from previous computation.
        Returns
        -------
        out:
            Gradient of the operand.
        """
        return [prev_grad * self.mask]


class BatchNorm(Operator):
    """
    Operator object for batch-normalization.
    """
    def compute(self, *var):  # var = [val, mean, stddev, eps]
        """
        Perform the batch-norm operation.
        The mean and stddev will be calculated in other module, so does the shifting.
        output = (value - mean) / (stddev + eps) # epsilon here to avoid divide by 0 error.

        Parameters
        ----------
        var: Variable..
            Value, mean, stddev and epsilon

        Returns
        -------
        out: Variable
            The batchnorm result.
        """
        return Variable((var[0].item - var[1]) / np.sqrt(var[2] + var[3]),
                        input_vars=[var[0], Variable(np.sqrt(var[2] + var[3]), no_grad=True)],
                        operator=self,
                        no_grad=True)

    def gradient(self, input_variables, prev_grad):
        """
        Calculate the gradient of batchnorm.
        Basically the batchnorm is the operation of:

        bn(x) = (x - mean) / (stddev + eps)

        thus:

        y = bn(x), z = f(y)

        dx/dz = (dy/dz) / (stddev + eps)

        Parameters
        ----------
        input_variables
        prev_grad

        Returns
        -------
        out:
            Gradient of the input.
        """
        return prev_grad / input_variables[1].item, 0


class GELU(Operator):
    """
    Estimating GELU Operator.
    """
    def compute(self, *var):
        """
        Estimates the gelu operation

        gelu(x) = 0.5x(1 + tanh(((2/pi)^-2) * (x + 0.044715 * x^3)))

        Parameters
        ----------
        var:
            input

        Returns
        -------
        out:
            GELU result
        """
        x = var[0].item
        return Variable(0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))),
                        input_vars=var,
                        operator=self,
                        no_grad=True)

    def gradient(self, input_variables, prev_grad):
        """
        Estimates the gradient of gelu operation.

        Parameters
        ----------
        input_variables:
            Input
        prev_grad:
            Gradient from previous operations.

        Returns
        -------
        out:
            Gradient of input.
        """
        x = input_variables[0].item
        return (0.5 * np.tanh(0.0356774 * np.power(x, 3) + 0.797885 * x) +
                (0.0535161 * np.power(x, 3) + 0.398942 * x) *
                np.power((1 / np.cosh(0.0356774 * np.power(x, 3) + 0.797885 * x)), 2) + 0.5) * prev_grad


def _p(var):
    """Caculate Bernoulli distribution"""
    return .5 * (1. + erf(var / np.sqrt(2.)))


class _GELU(Operator):
    """
    Accurate GELU computation.
    """
    def compute(self, *var):
        """
        Accurate GELU computation.

        GELU(x) = 0.5 * (1 + erf(x/(2^-2))) * x

        Parameters
        ----------
        var:
            Input

        Returns
        -------
        out:
            Result.
        """

        return Variable(var[0].item * _p(var[0].item),
                        input_vars=var,
                        operator=self,
                        no_grad=True)

    def gradient(self, input_variables, prev_grad):
        """
        Compute the accurate gradient of GELU operation.

        y = GELU(x), z = f(y)

        dx/dz = 0.5 * (1 + erf(x/(2^-1))) + x / (pi^-2) * e^(-(x^2)/2) * dy/dz

        Parameters
        ----------
        input_variables:
            Input of GELU
        prev_grad
            Gradient from previous operations
        Returns
        -------
        out:
            Gradient of input.
        """
        x = input_variables[0].item
        return (_p(x) + x / np.sqrt(np.pi) * np.exp(-np.power(x, 2) / 2)) * prev_grad


def relu(var: Variable) -> Variable:
    """
    Relu activation.

    x = 0 if x < 0 else 1

    Parameters
    ----------
    var:
        Input

    Returns
    -------
    out:
        Result of ReLU.
    """
    return ReLU()(var)


def softmax(var: Variable) -> Variable:
    """
    Softmax activation.

    x = e^x_i / sum(e^x)

    Parameters
    ----------
    var:
        Input

    Returns
    -------
    out:
        Result of Softmax.
    """
    return SoftMax()(var)


def batchNorm(x, mean, var, eps) -> Variable:
    """
    BatchNorm operation. Mean, stddev and eps must be defined beforehand.

    Parameters
    ----------
    x:
        Input
    mean:
        Mean of input.
    var:
        Stddev of input
    eps:
        a very small value to avoid divide by zero.

    Returns
    -------
    out:
        Result of BatchNorm
    """
    return BatchNorm()(x, mean, var, eps)


def gelu(var, estimate=True):
    """
    GELU activation.

    Parameters
    ----------
    var:
        Input
    estimate:
        If estimate the GELU result.

    Returns
    -------
    out:
        GELU result.
    """
    return GELU()(var) if estimate else _GELU()(var)


def do_nothing(var):
    return var
