import numpy as np
from dl.optimizer.Optim import Optim


class AdamOptimizer(Optim):

    def __init__(self, variables: list, lr=1e-2, momentum=(9e-1, 999e-3), eps=1e-8, weight_decay=0):
        """
        Adam optimizer. Based on SGDM and AdaGrad, Adam optimization is a stochastic gradient descent method that is
        based on adaptive estimation of first-order and second-order moments.

        gt = gt + Wt-1 * decayRate

        First order moment: a moving average of gradient.

        mt = b1 * mt-1 + (1 - b1) * gt

        Second order moment: a moving average of squared gradient.

        Vt = b2 * Vt-1 + (1 - b2) * gt^2

        Wt = Wt-1 - lr * mt / sqrt(Vt)

        Parameters
        ----------
        variables:
            Variables on watch. Each time the step() method was raise these variables will be updated.
        lr:
            Learning rate. The scale of variable updating.
        momentum:
            (First order momentum, second order momentum), to contol the moving average update scale.
        eps:
            A very small number to avoid divide by zero.
        weight_decay:
            Weight decay rate. Each round the variable was updated they will be reduced by this rate.
        """
        super().__init__(variables, lr)
        self.lr = lr
        self.beta1 = momentum[0]
        self.beta2 = momentum[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.velocity1 = [np.zeros_like(i.item) for i in self.variables]
        self.velocity2 = self.velocity1.copy()

    def step(self):
        """
        Calculate moving average of 1st order moment and 2st order moment, then update the watch variables.

        gt = gt + decayRate * Wt-1

        mt = b1 * mt-1 + (1 - b1) * gt

        Vt = b2 * Vt-1 + (1 - b2) * gt^2

        Wt = Wt-1 - lr * mt / sqrt(Vt)

        Returns
        -------
        None
        """
        for i in range(len(self.variables)):
            self.variables[i].grad += self.variables[i].item * self.weight_decay
            self.velocity1[i] = self.velocity1[i] * self.beta1 + (1 - self.beta1) * self.variables[i].grad
            self.velocity2[i] = self.velocity2[i] * self.beta2 + (1 - self.beta2) * np.power(self.variables[i].grad, 2)
            self.variables[i].item -= self.lr * self.velocity1[i] / (np.sqrt(self.velocity2[i]) + self.eps)
