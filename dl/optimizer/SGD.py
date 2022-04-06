from dl.optimizer import Optim
import numpy as np


class SGDOptimizer(Optim):

    def __init__(self, variables, momentum=None, lr=0.01, weight_decay=0):
        """
        Stochastic Gradient Descent optimizer, optional with momentum.

        Each round of updating, variables will be reduced by their gradient to the cost function scaled by lr.

        Wt = Wt-1 - lr * gt

        with momentum: The gradient will be replaced by moving average of the gradient.

        mt = b1 * mt-1 + (1 - b1) * gt

        Wt = Wt-1 - lr * mt

        If weight decay is defined, the gradient will be add by the variable * decayRate

        gt = gt + Wt-1 * decayRate

        Parameters
        ----------
        variables:
            Variables on watch
        momentum:
            1st order momentum moving average scale.
        lr:
            Learning rate.
        weight_decay:
            Weight decay Rate.
        """
        super().__init__(variables, lr)
        self._step_method = self._step
        self.momentum = momentum
        self.weight_decay = weight_decay
        if self.momentum is not None:  # If the momentum is defined, the behaviour will be different.
            self.velocity = [np.zeros_like(i.item) for i in self.variables]
            self._step_method = self._step_with_momentum

    def step(self):
        """
        Update the variables. Action will depend on whether the momentum is defined.

        Returns
        -------
        None
        """
        self._step_method()

    def _step(self):
        """
        Action without momentum.

        gt = gt + Wt-1 * decayRate

        Wt = Wt-1 - lr * mt

        Returns
        -------
        None
        """
        for variable in self.variables:
            variable.grad += variable.item * self.weight_decay
            variable.item -= variable.grad * self.lr

    def _step_with_momentum(self):
        """
        Action with momentum.

        gt = gt + Wt-1 * decayRate

        mt = b1 * mt-1 + (1 - b1) * gt

        Wt = Wt-1 - lr * mt

        Returns
        -------
        None
        """
        for i in range(len(self.variables)):
            self.variables[i].grad += self.variables[i].item * self.weight_decay
            self.velocity[i] = self.velocity[i] * self.momentum + (1 - self.momentum) * self.variables[i].grad
            self.variables[i].item -= self.velocity[i] * self.lr
