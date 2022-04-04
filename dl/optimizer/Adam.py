import numpy as np
from dl.optimizer.Optim import Optim


class AdamOptimizer(Optim):

    def __init__(self, variables: list, lr=1e-2, momentum=(9e-1, 999e-3), eps=1e-8, weight_decay=0):
        super().__init__(variables, lr)
        self.lr = lr
        self.beta1 = momentum[0]
        self.beta2 = momentum[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.velocity1 = [np.zeros_like(i.item) for i in self.variables]
        self.velocity2 = self.velocity1.copy()

    def step(self):
        for i in range(len(self.variables)):
            self.variables[i].grad += self.variables[i].item * self.weight_decay
            self.velocity1[i] = self.velocity1[i] * self.beta1 + (1 - self.beta1) * self.variables[i].grad
            self.velocity2[i] = self.velocity2[i] * self.beta2 + (1 - self.beta2) * np.power(self.variables[i].grad, 2)
            self.variables[i].item -= self.lr * self.velocity1[i] / (np.sqrt(self.velocity2[i]) + self.eps)
