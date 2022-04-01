import dl.graph.variable as variable
import numpy as np


class Operator(object):
    """
    calling operators between variables will instantiate new operator object
    to produce the new variable and indicate the compute process.
    """

    def __call__(self, *var):
        return self.compute(*var)

    def compute(self, *var):
        raise NotImplementedError

    def gradient(self, input_variables, prev_grad):
        """
        Call gradient() to compute the derivative of input variables to the source
        :param input_variables: the variable where the operator outputs
        :param prev_grad: the gradient of this variable to the source
        :return: the derivative of input variables to the source
        """
        raise NotImplementedError


class Add(Operator):

    def compute(self, *var):
        return variable.Variable(var[0].item + var[1].item,
                                 input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return prev_grad, prev_grad


class Mul(Operator):

    def compute(self, *var):
        return variable.Variable(var[0].item * var[1].item,
                                 input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return [input_variables[1].item * prev_grad, input_variables[0].item * prev_grad]


class MatMul(Operator):
    # Input features should be transposed
    def compute(self, *var):
        return variable.Variable(np.matmul(var[0].item, var[1].item),
                                 input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return np.matmul(prev_grad, input_variables[1].item.T), \
               np.matmul(input_variables[0].item.T, prev_grad)


class ReLU(Operator):
    def compute(self, *var):
        return variable.Variable(np.where(var[0].item > 0, var[0].item, 0),
                                 input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        grad = np.where(input_variables[0].item > 0, 1, 0) * prev_grad
        return [grad]


class SoftMax(Operator):
    # For mini-batch, the output layout will be same with the input
    def compute(self, *var):
        return variable.Variable(np.exp(var[0].item) / np.sum(np.exp(var[0].item), axis=0),
                                 input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return prev_grad  # No grad needed for output


class CrossEntropy(Operator):
    # y: var[0]
    # yhat: var[1]
    def compute(self, *var):
        return variable.Variable(
            -np.sum(var[0].item * np.log(var[1].item) / var[0].item.shape[1], axis=0),
            input_vars=var,
            operator=self,
            no_grad=True
        )

    def gradient(self, input_variables, prev_grad):
        return [0, input_variables[1].item - input_variables[0].item]


class Sub(Operator):

    def compute(self, *var):
        return variable.Variable(var[0].item - var[1].item,
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
        return variable.Variable(var[0].item * self.mask, input_vars=var, operator=self, no_grad=True)

    def gradient(self, input_variables, prev_grad):
        return prev_grad * self.mask


def relu(var):
    return ReLU()(var)
