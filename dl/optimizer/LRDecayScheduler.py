import numpy as np


class LRDecayScheduler:
    
    def __init__(self, optimizer, stride=None, magnification=1e-1, tolerance_round=10, expect_drop=1e-9):
        self.optimizer = optimizer
        self.stride = stride
        self.magnification = magnification
        self.tolerance_round = tolerance_round
        self.expected_drop = expect_drop
        self.round = 0
        self.prev_loss = 0xffffffff

    def step(self, loss):
        if self.prev_loss - np.mean(loss.item) > self.expected_drop:
            self.prev_loss = np.mean(loss.item)
            self.round = 0
        else:
            self.prev_loss = np.mean(loss.item)
            self.round += 1

        if self.round >= self.tolerance_round:
            if self.stride is not None:
                self.optimizer.lr -= self.stride
            else:
                self.optimizer.lr *= self.magnification
            self.round = 0
