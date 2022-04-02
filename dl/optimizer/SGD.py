from dl.optimizer import Optim


class SGDOptimizer(Optim):

    def __init__(self, variables, momentum=None, lr=0.01):
        super().__init__(variables, lr)
        self.momentum = momentum

    def step(self):
        for variable in self.variables:
            variable.item -= variable.grad * self.lr
