import dl.graph.op as op
import numpy as np


class Variable(object):
    """
    Object of all the numerical values in the computation to trace the operations and gradients.
    """
    def __init__(self, val: np.ndarray, input_vars=None, operator=None, no_grad=False) -> None:
        """
        Object of all the numerical values in the computation to trace the operations and gradients.

        Parameters
        ----------
        val:
            Numerical value this variable holds.
        input_vars:
            The input variable to result in this variable.
        operator:
            The operator object to result in this variable.
        no_grad:
            Is this variable need to be recorded its gradient.
        """
        self.item = val.astype(np.float64)  # value of this variable
        self.input_vars = input_vars  # the variables result in this variable
        self.operator = operator  # the operator to produce this variable
        self.shape = np.shape(val)
        self.grad = None  # this variable's gradient in the backprop
        self.no_grad = no_grad

    def __matmul__(self, other):
        if isinstance(other, Variable):
            return op.MatMul()(self, other)
        raise TypeError

    def __mul__(self, other):
        if isinstance(other, Variable):
            return op.Mul()(self, other)
        raise TypeError

    def __add__(self, other):
        if isinstance(other, Variable):
            return op.Add()(self, other)
        raise TypeError

    def __sub__(self, other):
        if isinstance(other, Variable):
            return op.Sub().compute(self, other)
        raise TypeError

    def __str__(self):
        return f"Variable({str(self.item)}, dtype={type(self.item)}, shape={self.shape})"

    def backward(self, prev_grad=None) -> None:
        """
        Backward Proporgation process.

        Each Variable will calculate the gradient of its input variables based on the previous gradient
        and call their backward methods. Each variable will update their gradient

        Parameters
        ----------
        prev_grad:
            Gradients from previous operation.

        Returns
        -------
        out:
            None
        """
        if prev_grad is None:  # prev_grad will be none if the variable is a leaf in the compuation graph.
            prev_grad = np.ones(self.shape)  # in that case, set the gradient to 1 since dx/dx = 1
        if not self.no_grad:
            if self.grad is None:
                self.grad = prev_grad.copy()
            else:
                self.grad += prev_grad
        if self.operator is not None:
            for var, grad in zip(self.input_vars, self.operator.gradient(self.input_vars, prev_grad)):
                var.backward(grad)

    __repr__ = __str__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
    __radd__ = __add__
