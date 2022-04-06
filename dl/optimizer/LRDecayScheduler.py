from dl.optimizer.Optim import Optim


class LRDecayScheduler:

    def __init__(self,
                 optimizer: Optim,
                 decay_mode="min",
                 factor=1e-1,
                 tolerance_round=100,
                 verbose=False,
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8):
        """
        LR decay Scheduler. After certain round of training, if the given metrics is not raising as expected, the
        scheduler will decrease the lr of given optimizer.

        Parameters
        ----------
        optimizer:
            Optimizer in use.
        decay_mode:
            "min" for a lower-better metric, "max" for a greater-better metric.
        factor:
            Decay rate of learning rate.
        tolerance_round:
            Max round of the score not become better before the lr decay.
        verbose:
            If ture, a notification will be printed after lr decay.
        cooldown:
            Rounds not measure metric before next lr decay.
        min_lr:
            Minimum lr that scheduler will dropped to.
        eps:
            If current - current lr * factor < eps, the lr decay will not be taken place.
        """
        self.optimizer = optimizer
        self.stride = factor
        self.factor = factor
        self.tolerance_round = tolerance_round
        if decay_mode not in ["max", "min"]:
            raise ValueError("Decay mode should either be 'max' or 'min'")
        self.max_decay = decay_mode == "max"
        self.verbose = verbose
        self.cooldown = cooldown
        self.is_cooling_down = False
        self.min_lr = min_lr
        self.eps = eps
        self.round = 0
        self.best_score = 0 if self.max_decay else 0xffffffff

    def step(self, val: float):
        """
        Evaluate the metric value.

        Each time this method was called, the scheduler will see if this number is better than previous best score. If
        it is, zero the counter and record new best score.

        Parameters
        ----------
        val:
            Metric. Like loss or accuracy.

        Returns
        -------
        None.
        """
        self.round += 1
        if self.is_cooling_down:
            if self.round >= self.cooldown:
                self.is_cooling_down = False
                self.round = 0
        else:
            if self.max_decay:
                if self.best_score < val:
                    self.round = 0
                    self.best_score = val
            else:
                if self.best_score > val:
                    self.round = 0
                    self.best_score = val
            if self.round >= self.tolerance_round:
                if self.optimizer.lr > self.min_lr and self.optimizer.lr * (1 - self.factor) > self.eps:
                    if self.verbose:
                        print(f"Learning rate dropped from {self.optimizer.lr} to {self.optimizer.lr * self.factor}")
                    self.optimizer.lr *= self.factor
                self.round = 0
                self.is_cooling_down = True
