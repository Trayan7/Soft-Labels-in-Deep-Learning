import numpy as np
import wandb
from keras.callbacks import Callback

from evaluation.metrics import center_diff_loss_func, second_best_accuracy_func


class IntervalEvaluation(Callback):
    '''
    Handles the evaluation of computationally expensive metrics
    on epoch end after the set number of epochs pass for soft labels.
    '''
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            pred = self.model.predict(self.X_val, verbose=0)
            pred_norm = []
            for elem in pred:
                cur = elem.copy()
                # Set negative values to 0
                cur[cur < 0] = 0
                pred_norm.append(elem / elem.sum())
            np_y_val = np.array(self.y_val)
            np_x_val = np.array(self.X_val)
            center_diff = center_diff_loss_func(pred_norm, np_y_val, 10)
            scnd_acc = second_best_accuracy_func(pred, np_y_val)
            wandb.log({'second_best_accuracy': scnd_acc, 'center_diff': center_diff, 'epoch': epoch})

class IntervalEvaluationHard(Callback):
    '''
    Handles the evaluation of computationally expensive metrics
    on epoch end after the set number of epochs pass for hard labels.
    '''
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            pred = self.model.predict(self.X_val, verbose=0)
            pred_norm = []
            for elem in pred:
                cur = elem.copy()
                # Set negative values to 0
                cur[cur < 0] = 0
                pred_norm.append(elem / elem.sum())
            np_y_val = np.array(self.y_val)
            center_diff = center_diff_loss_func(pred_norm, np_y_val, 10)
            wandb.log({'center_diff': center_diff, 'epoch': epoch})
