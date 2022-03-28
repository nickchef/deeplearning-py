from typing import Any
import dl.graph.op as op


class Variable(object):

    def __init__(self, val: Any, input_vars=None, operator=None):
        self.item = val  # value of this variable
        self.input_vars = input_vars  # the variables result in this variable
        self.operator = operator  # the operator to produce this variable
        # self.name = name  # name of this variable to debug
        # self.no_grad = no_grad  # if this variable need to compute gradient
        self.grad = 0  # this variable's gradient in the backprop
        self.name = f"Variable({str(self.item)})"
        # self.const = const  # if one of this variable's input is a constant value

    def __matmul__(self, other):
        return op.MatMul()(self, other)

    def __mul__(self, other):
        return op.Mul()(self, other)

    def __add__(self, other):
        if isinstance(other, Variable):
            return op.Add()(self, other)

    # def __sub__(self, other):
    #     return Sub().compute(self, other)

    def __str__(self):
        return self.name

    def backward(self, prev_grad=1):
        self.grad += prev_grad
        if self.operator is not None:
            for var, grad in zip(self.input_vars, self.operator.gradient(self.input_vars, prev_grad)):
                var.backward(grad)

    __repr__ = __str__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
    __radd__ = __add__
    # __rsub__ = __sub__
