from dl.nn import Module
import dl.graph.op as OP


class ReLULayer(Module):

    def __init__(self):
        super().__init__()
        self.op = OP.ReLU()

    def forward(self, x):
        return self.op(x)
