
class Optim:

    def __init__(self, variables: list, lr: float):
        """
        Basic class of optimizer object.

        Parameters
        ----------
        variables:
            Variables on watch.
        lr:
            Learning rate.
        """
        self.variables = variables
        self.lr = lr

    def zero_grad(self):
        """
        Set the gradient of all variables on watch to None.

        Returns
        -------
        None
        """
        for variable in self.variables:
            variable.grad = None

    def step(self):
        """
        Any customized optimizer should override this method to define their behaviour.

        Returns
        -------
        None
        """
        raise NotImplementedError
