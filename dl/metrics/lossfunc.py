from dl.graph.variable import Variable
import dl.graph.op as OP


class LossFunction:
    """
    Base class of loss functions.
    """
    def __call__(self, y, yhat) -> Variable:
        """
        Compute the loss based on prediction and ground truth.

        Parameters
        ----------
        y:
            Prediction
        yhat:
            Ground Truth
        Returns
        -------
        out:
            Loss value of prediction.
        """
        return self.compute(y, yhat)

    def compute(self, y, yhat) -> Variable:
        raise NotImplementedError


class CrossEntropyLoss(LossFunction):

    def __init__(self):
        """
        Cross Entropy object.

        See details at dl.graph.op.CrossEntropy

        """
        self.op = OP.CrossEntropy()

    def compute(self, y, yhat) -> Variable:
        return self.op(y, yhat)
