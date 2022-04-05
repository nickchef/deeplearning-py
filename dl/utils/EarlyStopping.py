from typing import Callable
import dl


class CatchBestModel:

    def __init__(self, model: dl.nn.Module, mode="max"):
        self.model = model
        self.best_Score = 0 if mode == "max" else 0xffffffff
        self.mode = mode
        self.best_state = None

    def step(self, val: float):
        if (self.mode == "max" and val > self.best_Score) or (self.mode == "min" and val < self.best_Score):
            self.best_Score = val
            self.best_state = self.model.save_parameters()

    def get_best_model(self):
        self.model.load_parameters(self.best_state)
        return self.model


class EarlyStoppingPipe:

    def __init__(self, model: dl.nn.Module,
                 train: Callable,
                 max_epoch: int,
                 params=None,
                 mode="max",
                 tolerance_round=10,
                 value_on_watch=None) -> None:
        self.model = model
        self.best_Score = 0 if mode == "max" else 0xffffffff
        self.mode = mode
        self.best_state = None
        self.round = 0
        self.tolerance_round = tolerance_round
        self.max_epoch = max_epoch
        self.train = train
        self.params = params
        self.data = []
        self.value_on_watch = value_on_watch

    def run(self):
        for i in range(self.max_epoch):
            print(f"Epoch {i+1}: ", end="")
            if self.params is not None:
                ret = self.train(*self.params)
            else:
                ret = self.train()
            self.data.append(ret)
            val = ret if self.value_on_watch is None else ret[self.value_on_watch]
            if (self.mode == "max" and val > self.best_Score) or (self.mode == "min" and val < self.best_Score):
                self.best_Score = val
                self.best_state = self.model.save_parameters()
                self.round = 0
            else:
                self.round += 1
            if self.round > self.tolerance_round:
                print(f"EarlyStopped with best score {self.best_Score} at epoch {i+1}")
                break
        return self.get_best_model()

    def get_best_model(self):
        if self.best_state is not None:
            self.model.load_parameters(self.best_state)
            return self.model
        else:
            raise RuntimeError("Training has not been processed!")

    def get_evaluation_data(self):
        return self.data.copy()
