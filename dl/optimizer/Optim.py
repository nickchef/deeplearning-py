
class Optim:

    def __init__(self, variables: list, lr: float):
        self.variables = variables
        self.lr = lr

    def zero_grad(self):
        for variable in self.variables:
            variable.grad = None

    def step(self):
        raise NotImplementedError
