from dl.nn.Module import Module
import dl.graph.op as OP
from dl.graph import variable


class DropoutLayer(Module):
    """
    Dropout layer object.
    """
    def __init__(self, rate: float):
        """
        Dropout layer object.

        Parameters
        ----------
        rate:
            Dropout rate.
        """
        super().__init__()
        self.op = OP.Dropout(rate)

    def forward(self, x) -> variable.Variable:
        """
        Process the dropout operation.
        See details at dl.graph.op.Dropout

        Parameters
        ----------
        x:
            Input

        Returns
        -------
        out:
            output
        """
        return self.op(x)

    def eval(self):
        """
        Set the layer to evaluation mode. in this mode, dropout will not be performed.

        Returns
        -------
        out:
            None
        """
        self.op.eval = True

    def train(self):
        """
        Set the layer to evaluation mode. in this mode, dropout will be performed.

        Returns
        -------
        out:
            None
        """
        self.op.eval = False
