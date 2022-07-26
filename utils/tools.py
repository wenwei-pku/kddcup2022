import numpy as np
import torch
import pandas as pd


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args['lradj'] == 'type1':
        lr_adjust = {epoch: args['learning_rate'] * (0.5 ** ((epoch - 1) // 1))}
    elif args['lradj'] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        lr_adjust = {epoch: args['learning_rate']}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean




def iqr_outliers_mask(x, coef=1.5):
    """

    Args:
        x: np.array like
        coef: float.
    Returns: boolean array. False as normal values, True for outliers

    """
    if len(x) <= 1:
        return np.zeros_like(x, dtype=bool)

    upper = np.nanquantile(x, 0.75)
    lower = np.quantile(x, 0.25)
    iqr = (upper - lower) * coef
    outliers_mask = np.logical_or(x < (lower - iqr), x > (upper + iqr))

    return outliers_mask



def abnormal_mask(df):
    """
    Rules
    1. Zero values
    $Patv_t$=0 if $Patv_t$ < 0

    2. Missing values
    $Patv_t$ doesn't exist.

    3. Unknown values
    Patv < 0 and Wspd > 2.5, or any of $Pab_i$ > 89°, i $\in$ [1, 2, 3].

    4. Abnormal values
    $Ndir_t$ > 720° or $Ndir_t$ < -720°, or $Wdir_t$ > 180° or $Wdir_t$ < -180° .


    Args:
        df: pandas DataFrame. Raw data from each turbine only.

    Returns:
        masks

    """

    abnormals_1 = (df['Wspd'] > 2.5) & (df['Patv'] <= 0)
    abnormals_2 = df[['Pab1', 'Pab2', 'Pab3']].max(1) > 89
    abnormals_3 = (df['Ndir'] < -720) | (df['Ndir'] > 720)
    abnormals_4 = (df['Wdir'] < -180) | (df['Wdir'] > 180)

    missing_values = df.isna().mean(1).astype(bool) # mask all missing values

    abnormals = abnormals_1 | abnormals_2 | abnormals_3 | abnormals_4 | missing_values

    return abnormals