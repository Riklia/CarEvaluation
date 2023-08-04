import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        """
        Stop training when a monitored metric has stopped improving.
        Args:
            patience: (int) Number of epochs with no improvement after which training will be stopped
            mode: (string) `min` or `max`. Direction of the loss optimization
            delta: (float) An absolute loss change of less than delta will count as no improvement
        """

        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        """
        Args:
            epoch_score: (float) metric that is beeing optimized
            model: (torch.nn.Module) the neural network
            model_path: (string) folder where parameters are to be saved
        """

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('\nEarlyStopping counter: {} out of {}'.format(self.counter,
                                                               self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('\nValidation score improved ({} --> {}). Saving '
                  'model!'.format(
                self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score
