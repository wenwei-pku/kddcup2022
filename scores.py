import pandas as pd
import numpy as np
import os
import math
import joblib

from data_provider.data_loader import *
from prepare import prep_env
from data_provider.data_loader import DatasetTree
from utils.metrics import *

config = prep_env()

data_provider = DatasetTree(
    Path(config['data_path'], config['filename']),
    train_days=config['train_days'],
    val_days=config['val_days'],
    test_days=config['test_days'],
    input_len=config['input_len'],
    pred_len=config['output_len']
)

maes = []
rmses = []
scores = []

for turb_id in range(1, 5):
    train, val, test, test_raw = data_provider.train_test_split(turb_id=turb_id)
    test_raw = test_raw[test_raw.Day >= 231]

    test = []
    test_df = []
    for i in range(len(test_raw) - config['output_len']):
        test.append(test_raw['Patv'].values[i:i + config['output_len']])
        test_df.append(test_raw[i:i + config['output_len']])
    test = np.array(test)

    # ypred = np.load(f'preds/pred_{turb_id}.npy')
    ypred = np.load('ypred.npy')
    # ypred = np.expand_dims(ypred, axis=-1)
    ypred=np.clip(ypred, 0, 1300)
    test = np.expand_dims(test, axis=-1)

    mae = []
    rmse = []
    score = []
    # for j in range(ypred.shape[0] - self.config['output_len']):
    for j in range(ypred.shape[0]):
        cur_mae, cur_rmse = turbine_scores(ypred[j, :, :],
                                           test[j, :, :],
                                           test_raw[j: j + config['output_len']],
                                           # examine_len=int(1e5))
                                           examine_len=config['output_len'])
        mae.append(cur_mae)
        rmse.append(cur_rmse)
        score.append((cur_mae + cur_rmse) / 2)

    maes.append(np.mean(mae))
    rmses.append(np.mean(rmse))
    scores.append(np.mean(score))
