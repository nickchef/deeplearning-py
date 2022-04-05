
class Module:

    def __init__(self):
        self.op = None
        self.variables = None
        self.children = None

    def __call__(self, x):
        return self.forward(x)

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

    def save_parameters(self):
        param = self.get_parameters()
        saved_parameter = []
        for i in param:
            saved_parameter.append(i.item.copy())
        return saved_parameter

    def load_parameters(self, params: list):
        param = self.get_parameters()
        for i in range(len(param)):
            param[i].item = params[i].copy()

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

    def forward(self, x):
        raise NotImplementedError

    def apply(self, func):
        if self.children is None:
            self._get_sub_modules()
        for i in self.children:
            func(i)
