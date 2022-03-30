import numpy as np

from dl import Module
from dl.graph import variable


class DenseLayer(Module):

    def __init__(self, input_dims, output_dims) -> None:
        super().__init__()
        self.weight = variable.Variable(
            np.array([[1 for _ in range(input_dims * output_dims)]]).reshape(output_dims, input_dims)
        )  # shape(output_dim, input_dim)
        self.bias = variable.Variable(np.array([[0 for _ in range(output_dims)]]))  # shape(1, output_dim)
        self.variables = [self.weight, self.bias]

    def forward(self, x) -> variable.Variable:
        return (self.weight @ x) + self.bias
