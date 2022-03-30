
class Module:

    def __init__(self):
        self.op = None
        self.variables = None
        self.children = None

    def __call__(self, *args):
        self.forward(*args)

    def _get_sub_modules(self):
        self.children = []
        for attr in vars(self).values():
            if isinstance(attr, Module):
                self.children.append(attr)

    def get_parameters(self):
        if self.children is None:
            self._get_sub_modules()
        parameters = []
        for i in self.children:
            if i.variables is not None:
                parameters += i.variables
        return parameters

    def train(self):
        if self.children is None:
            self._get_sub_modules()
        for i in self.children:
            i.train()

    def eval(self):
        if self.children is None:
            self._get_sub_modules()
        for i in self.children:
            i.eval()

    def forward(self, *args):
        raise NotImplementedError

    def apply(self, func):
        if self.children is None:
            self._get_sub_modules()
        for i in self.children:
            func(i)
