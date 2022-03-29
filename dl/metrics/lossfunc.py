import numpy as np
from dl.graph.variable import Variable
from dl.graph.op import SoftMax, Sub


class LossFunction:

    def __call__(self, y, yhat):
        return self.compute(y, yhat)

    def compute(self, y, yhat) -> Variable:
        raise NotImplementedError


class SoftMaxLoss(LossFunction):

    def compute(self, y, yhat) -> Variable:
        pred = SoftMax()(y)
        return y - yhat
