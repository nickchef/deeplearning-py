from typing import Callable
import dl


class CatchBestModel:

    def __init__(self, model: dl.nn.Module, mode="max"):
        """
        Watch on the model during training process by the given metric, catch the state of the best model.

        Parameters
        ----------
        model:
            Model on watch.
        mode:
            "min" for a lower-better metric, "max" for a greater-better metric.
        """
        self.model = model
        self.best_Score = 0 if mode == "max" else 0xffffffff
        self.mode = mode
        self.best_state = None

    def step(self, val: float):
        """
        After each round of training, call this method to evaluate the performance.

        Parameters
        ----------
        val:
            Metrics, e.g. accuracy or loss.

        Returns
        -------
        None
        """
        if (self.mode == "max" and val > self.best_Score) or (self.mode == "min" and val < self.best_Score):
            self.best_Score = val
            self.best_state = self.model.save_parameters()

    def get_best_model(self) -> dl.nn.Module:
        """
        Get the best model during the training. This method will override the state of trained model, replaced by
        the best state.

        Returns
        -------
        Model:
            The best model catched in training.
        """
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
        """
        A pipe to train the model and stop when the metric is not getting better in certain epoches.

        Parameters
        ----------
        model:
            Model to be trained.
        train:
            A method to train the model. Metrics must be returned from this method. This method must be for single
            epoch.
        max_epoch:
            Max epoches to train. The process will be stopped after this epoch.
        params:
            Parameters to be passed into the training method.
        mode:
            "min" for a lower-better metric, "max" for a greater-better metric.
        tolerance_round:
            The round limit for metrics to become better.
        value_on_watch:
            The index of mesurement in the returned value of training method. Set to None if the method return
            a single value.
        """
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
        self.best_round = 0

    def run(self):
        """
        Begin training.

        Returns
        -------
        model:
            The best model catched in the training process.
        """
        for i in range(self.max_epoch):
            # Begin epoch
            print(f"Epoch {i+1}: ", end="")
            if self.params is not None:
                ret = self.train(*self.params)
            else:
                ret = self.train()
            self.data.append(ret)  # record all the returned value
            val = ret if self.value_on_watch is None else ret[self.value_on_watch]
            # mesure the value
            if (self.mode == "max" and val > self.best_Score) or (self.mode == "min" and val < self.best_Score):
                self.best_Score = val
                self.best_state = self.model.save_parameters()
                self.best_round = i
                self.round = 0
            else:
                self.round += 1
            if self.round > self.tolerance_round:
                print(f"EarlyStopped with best score {self.best_Score} at epoch {self.best_round + 1}")
                break
        return self.get_best_model()

    def get_best_model(self):
        """
        Get the best model catched in the training process.

        Returns
        -------
        model:
            The best model catched in the training process.
        """
        if self.best_state is not None:
            self.model.load_parameters(self.best_state)
            return self.model
        else:
            raise RuntimeError("Training has not been processed!")

    def get_evaluation_data(self) -> list:
        """
        Get the returned data of each rounds during training.

        Returns
        -------
        data:
            Returned data of each rounds during training.
        """
        return self.data.copy()
