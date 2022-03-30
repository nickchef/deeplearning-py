from dl.graph.variable import Variable
import dl.graph.op as OP


class LossFunction:

    def __call__(self, y, yhat):
        return self.compute(y, yhat)

    def compute(self, y, yhat) -> Variable:
        raise NotImplementedError


class CrossEntropyLoss(LossFunction):

    def __init__(self):
        self.op = OP.CrossEntropy()

    def compute(self, y, yhat) -> Variable:
        return self.op(y, yhat)
