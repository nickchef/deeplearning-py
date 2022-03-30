from typing import Any
import dl.graph.op as op
import numpy as np


class Variable(object):

    def __init__(self, val: Any, input_vars=None, operator=None):
        self.item = val  # value of this variable
        self.input_vars = input_vars  # the variables result in this variable
        self.operator = operator  # the operator to produce this variable
        self.grad = None  # this variable's gradient in the backprop
        self.shape = np.shape(self.item)

    def __matmul__(self, other):
        return op.MatMul()(self, other)

    def __mul__(self, other):
        return op.Mul()(self, other)

    def __add__(self, other):
        if isinstance(other, Variable):
            return op.Add()(self, other)

    def __sub__(self, other):
        return op.Sub().compute(self, other)

    def __str__(self):
        return f"Variable({str(self.item)}, dtype={type(self.item)}, shape={self.shape})"

    def backward(self, prev_grad=None):
        if prev_grad is None:
            prev_grad = np.ones(self.shape)
        if self.grad is None:
            self.grad = np.zeros(self.shape)
        self.grad += prev_grad
        if self.operator is not None:
            for var, grad in zip(self.input_vars, self.operator.gradient(self.input_vars, prev_grad)):
                var.backward(grad)

    __repr__ = __str__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
    __radd__ = __add__
