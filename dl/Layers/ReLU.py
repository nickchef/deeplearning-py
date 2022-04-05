from dl.nn import Module
import dl.graph.op as OP


class ReLULayer(Module):
    """
    Relu layer object.
    """
    def __init__(self):
        """
        ReLU layer object.
        """
        super().__init__()
        self.op = OP.ReLU()

    def forward(self, x):
        """
        Process ReLU operation.
        See details at dl.graph.op.ReLU

        Parameters
        ----------
        x:
            Input

        Returns
        -------
        out:
            ReLU result.
        """
        return self.op(x)
