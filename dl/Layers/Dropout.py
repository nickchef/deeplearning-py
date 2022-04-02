from dl.nn.Module import Module
import dl.graph.op as OP
from dl.graph import variable


class DropoutLayer(Module):

    def __init__(self, rate):
        super().__init__()
        self.op = OP.Dropout(rate)

    def forward(self, x) -> variable.Variable:
        return self.op(x)

    def eval(self):
        self.op.eval = True

    def train(self):
        self.op.eval = False
