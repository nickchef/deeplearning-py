from dl.nn.Module import Module
from dl.graph import variable
from dl.graph.op import do_nothing, Dropout
from dl.nn.init import xavier_uniform_init


class DenseLayer(Module):
    def __init__(self, input_dims, output_dims, **kwargs) -> None:
        super().__init__()
        initializer = kwargs["initializer"] if "initializer" in kwargs.keys() else xavier_uniform_init
        i_weight, i_bias = initializer(input_dims, output_dims)
        self.weight = variable.Variable(i_weight)
        self.bias = variable.Variable(i_bias)  # shape(1, output_dim)
        self.variables = [self.weight, self.bias]

        self.activation = kwargs["activation"] if "activation" in kwargs.keys() else do_nothing
        self.dropout = Dropout(kwargs["dropout"]) if "dropout" in kwargs.keys() else do_nothing

    def forward(self, x):
        return self.dropout(self.activation((self.weight @ x) + self.bias))

    def eval(self):
        if isinstance(self.dropout, Dropout):
            self.dropout.eval = True

    def train(self):
        if isinstance(self.dropout, Dropout):
            self.dropout.eval = False

