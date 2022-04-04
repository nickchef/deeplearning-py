from dl.optimizer import Optim
import numpy as np


class SGDOptimizer(Optim):

    def __init__(self, variables, momentum=None, lr=0.01, weight_decay=0):
        super().__init__(variables, lr)
        self._step_method = self._step
        self.momentum = momentum
        self.weight_decay = weight_decay
        if self.momentum is not None:
            self.velocity = [np.zeros_like(i.item) for i in self.variables]
            # self.step = self._init_velocity
            self._step_method = self._step_with_momentum

    def step(self):
        self._step_method()

    def _step(self):
        for variable in self.variables:
            variable.grad += variable.item * self.weight_decay
            variable.item -= variable.grad * self.lr

    def _step_with_momentum(self):
        for i in range(len(self.variables)):
            self.variables[i].grad += self.variables[i].item * self.weight_decay
            self.velocity[i] = self.velocity[i] * self.momentum + (1 - self.momentum) * self.variables[i].grad
            self.variables[i].item -= self.velocity[i] * self.lr
