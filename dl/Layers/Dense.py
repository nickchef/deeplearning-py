from dl.nn.Module import Module
from dl.graph.variable import Variable
from dl.graph.op import do_nothing, Dropout
from dl.nn.init import xavier_uniform_init


class DenseLayer(Module):
    """
    Full-connected layer object.
    """
    def __init__(self, input_dims: int, output_dims: int, **kwargs) -> None:
        """
        Full-connected layer object.

        Parameters
        ----------
        input_dims:
            input dimension
        output_dims:
            output dimension, nueron number of this layer
        kwargs:
            initializer: initialize function from dl.nn.init to initialize the weight and bias.

            activation: activation function of this layer.

            dropout: Dropout rate for this layer.
        """
        super().__init__()
        initializer = kwargs["initializer"] if "initializer" in kwargs.keys() else xavier_uniform_init
        i_weight, i_bias = initializer(input_dims, output_dims)
        self.weight = Variable(i_weight)
        self.bias = Variable(i_bias)
        self.variables = [self.weight, self.bias]

        self.activation = kwargs["activation"] if "activation" in kwargs.keys() else do_nothing
        self.dropout = Dropout(kwargs["dropout"]) if "dropout" in kwargs.keys() else do_nothing

    def forward(self, x: Variable) -> Variable:
        """
        Forward proporgation.

        out = w @ xT + b

        Parameters
        ----------
        x:
            Input

        Returns
        -------
        out:
            Forward result.
        """
        return self.dropout(self.activation((self.weight @ x) + self.bias))

    def eval(self):
        """
        Set the layer to evaluation mode. in this mode, dropout will not be performed.

        Returns
        -------
        out:
            None
        """
        if isinstance(self.dropout, Dropout):
            self.dropout.eval = True

    def train(self):
        """
        Set the layer to training mode. in this mode, dropout will be performed.

        Returns
        -------
        out:
            None
        """
        if isinstance(self.dropout, Dropout):
            self.dropout.eval = False

