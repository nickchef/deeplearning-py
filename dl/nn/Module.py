from typing import Callable


class Module:
    """
    Basic class of modules in neural network. To customize a module, inherit this class and override the forward()
    method to define the module behaviour. A submodule can also be defined as the member variable.
    """
    def __init__(self):
        self.op = None  # operator
        self.variables = None  # variable
        self.children = None  # submodules

    def __call__(self, x):
        return self.forward(x)

    def _get_sub_modules(self):
        """
        Detect any submodule of this module, collect reference into self.children. This method will triggered when
        the get_parameter() method was called.

        Returns
        -------
        None
        """
        self.children = []
        for attr in vars(self).values():
            if isinstance(attr, Module):
                self.children.append(attr)

    def get_parameters(self):
        """
        Get the reference of parameters of this module and all of its submodules.

        Returns
        -------
        Parameters:
            All parameters of this module and all of its submodules.
        """
        if self.children is None:
            self._get_sub_modules()
        parameters = []
        for i in self.children:
            if i.variables is not None:
                parameters += i.variables
        return parameters

    def save_parameters(self):
        """
        Get the copy of parameters of this module and all of its submodules.

        Returns
        -------
        Parameters:
            Copies of parameters of this module and all of its submodules.
        """
        param = self.get_parameters()
        saved_parameter = []
        for i in param:
            saved_parameter.append(i.item.copy())
        return saved_parameter

    def load_parameters(self, params: list) -> None:
        """
        Load the parameters.

        Parameters
        ----------
        params:
            Parameters saved previously.

        Returns
        -------
        None
        """
        param = self.get_parameters()
        for i in range(len(param)):
            param[i].item = params[i].copy()

    def train(self):
        """
        Set this module and all submodules to training mode.

        Returns
        -------
        None
        """
        if self.children is None:
            self._get_sub_modules()
        for i in self.children:
            i.train()

    def eval(self):
        """
        Set this module and all submodules to evaluation mode.

        Returns
        -------
        None
        """
        if self.children is None:
            self._get_sub_modules()
        for i in self.children:
            i.eval()

    def forward(self, x):
        """
        Customized module should override this function to define the module behaviour.

        Parameters
        ----------
        x:
            Input

        Returns
        -------
        out:
            Output
        """
        raise NotImplementedError

    def apply(self, func: Callable) -> None:
        """
        Apply given function to submodules

        Parameters
        ----------
        func: Callable
            Function which arguments must be module.
        Returns
        -------
        None
        """
        if self.children is None:
            self._get_sub_modules()
        for i in self.children:
            func(i)
