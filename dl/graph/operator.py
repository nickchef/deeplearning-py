from abc import abstractmethod
from variable import Variable
import numpy as np


class Operator(object):
    def __call__(self):
        new_variable = Variable()
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
        return var[0] + var[1]

    def gradient(self, variable, output_grad):
        return output_grad


class Mul(Operator):

    def compute(self, *var):
        return var[0] * var[1]

    def gradient(self, variable, output_grad):
        return variable.input[1]*output_grad, variable[0]*output_grad


class Log(Operator):

    def compute(self, *var):
        return np.log(var[0])

    def gradient(self, variable, output_grad):
        return


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
