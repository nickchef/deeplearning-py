from dl.optimizer.Optim import Optim


class LRDecayScheduler:

    def __init__(self,
                 optimizer: Optim,
                 decay_mode="min",
                 factor=1e-1,
                 tolerance_round=10,
                 verbose=False,
                 cooldown=0,
                 min_lr=0,
                 threshold=1e-4,
                 eps=1e-8):

        self.optimizer = optimizer
        self.stride = factor
        self.factor = factor
        self.tolerance_round = tolerance_round
        self.threshold = threshold
        if decay_mode not in ["max", "min"]:
            raise ValueError("Decay mode should either be 'max' or 'min'")
        self.max_decay = decay_mode == "max"
        self.verbose = verbose
        self.cooldown = cooldown
        self.is_cooling_down = False
        self.min_lr = min_lr
        self.eps = eps
        self.round = 0
        self.prev_number = 0 if self.max_decay else 0xffffffff

    def step(self, val: float):
        self.round += 1
        if self.is_cooling_down:
            if self.round >= self.cooldown:
                self.is_cooling_down = False
                self.round = 0
        else:
            if self.max_decay:
                if val - self.prev_number > self.threshold:
                    self.round = 0
            else:
                if self.prev_number - val > self.threshold:
                    self.round = 0
            if self.round >= self.tolerance_round:
                if self.optimizer.lr > self.min_lr and self.optimizer.lr * (1 - self.factor) > self.eps:
                    if self.verbose:
                        print(f"Learning rate dropped from {self.optimizer.lr} to {self.optimizer.lr * self.factor}")
                    self.optimizer.lr *= self.factor
                self.round = 0
                self.is_cooling_down = True
        self.prev_number = val
