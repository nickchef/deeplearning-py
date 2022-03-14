from typing import Any
from collections import Iterable
from .op import *


class Variable(object):

    def __init__(self, val: Any, input_vars: Iterable,
                 operator: Operator, name: str, no_grad=False, grad=0):
        self.val = val  # value of this variable
        self.input_vars = input_vars  # the variables result in this variable
        self.operator = operator  # the operator to produce this variable
        # self.const = const
        self.name = name  # name of this variable to debug
        self.no_grad = no_grad  # if this variable need to compute gradient
        self.grad = grad  # this variable's gradient in the backprop

    def __mul__(self, other) -> Variable:
        return Mul(self, other)

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass
