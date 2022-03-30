from dl import Module
from dl.graph import variable


class BatchNormLayer(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x) -> variable.Variable:
        return variable.Variable(0)
