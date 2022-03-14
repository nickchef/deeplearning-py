from abc import abstractmethod
from variable import Variable
import numpy as np


class Operator(object):
    """
    calling operators between variables will instantiate new operator object
    to produce the new variable and indicate the compute process.
    """

    def __call__(self, *var):
        new_variable = self.compute(*var)
        new_variable.operator = self
        return new_variable

    @abstractmethod
    def compute(self, *var):
        pass

    @abstractmethod
    def gradient(self, variable, output_grad):
        """
        Call gradient() to compute the derivative of input variables to the source
        :param variable: the variable where the operator outputs
        :param output_grad: the gradient of this variable to the source
        :return: the derivative of input variables to the source
        """
        pass


class Add(Operator):

    def compute(self, *var):
        return Variable(var[0].val + var[1].val, var, self,
                        "{} + {}".format(var[0].name, var[1].name))

    def gradient(self, variable, output_grad):
        return [output_grad, output_grad]


class Mul(Operator):

    def compute(self, *var):
        return Variable(np.matmul(var[0].val, var[1].val), var, self,
                        "({}) * ({})".format(var[0].name, var[1].name))

    def gradient(self, variable, output_grad):
        return [variable.input[1]*output_grad, variable[0]*output_grad]


class Log(Operator):
    # log e
    def compute(self, *var):
        return Variable(np.log(var[0].val), var, self,
                        "log({})".format(var[0].name, var[1].name))

    def gradient(self, variable, output_grad):
        return [(1/variable[0])*output_grad]


class ReLU(Operator):
    def compute(self, *var):
        return Variable(np.where(var[0].val > 0, var[0].val, 0),
                        var,
                        self,
                        "relu({})".format(var[0].val))

    def gradient(self, variable, output_grad):
        return np.where(variable.input[0].val > 0, 1, 0) * output_grad


class Divide(Operator):

    def compute(self, *var):
        return var[0] / var[1]

    def gradient(self, variable, output_grad):
        return 


class PlaceHolder(Operator):

    def compute(self, *var):
        pass

    def gradient(self, variable, output_grad):
        pass
