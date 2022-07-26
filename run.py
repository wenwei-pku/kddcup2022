import torch
import numpy as np
import pandas as pd
import argparse
import math
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from yaml import parse

from prepare import prep_env
from exp.exp_lgb import ExpLgb
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='n_jobs', type=int, default=8, help="CV N jobs")
parser.add_argument('-i', dest='inner_nfold', type=int, default=2, help="Number of inner folds")
parser.add_argument('-t', dest='model_type', type=str, default='lgb', help="Model type")
parser.add_argument('-r', dest='seed', type=int, default=(1 << 31) - 1, help='Random seed.')

parser.add_argument('--rf-n', dest='rf_n_estimators', type=int, default=100, help='RF hyperparameter.')
parser.add_argument('--rf-l', dest='rf_min_samples_leaf', type=int, default=3, help='RF hyperparameter.')
parser.add_argument('--rf-s', dest='rf_max_samples', type=float, default=None, help='RF hyperparameter.')
parser.add_argument('--rf-f', dest='rf_max_features', type=float, default=1/math.exp(1), help='RF hyperparameter.')
parser.add_argument('--rf-cv', dest='rf_cv', action='store_true', help='Tune RF hyperparameters.')
parser.add_argument('--lgb-n', dest='lgb_n_estimators', type=int, default=512, help='LGB hyperparameter.')
parser.add_argument('--lgb-l', dest='lgb_num_leaves', type=int, default=127, help='LGB hyperparameter.')
parser.add_argument('--lgb-lr', dest='lgb_learning_rate', type=float, default=0.025, help='LGB hyperparameter.')
parser.add_argument('--lgb-f', dest='lgb_feature_frac', type=float, default=1/math.exp(1), help='LGB hyperparameter.')
parser.add_argument('--lgb-sq', dest='lgb_subsample_freq', type=int, default=0, help='LGB hyperparameter.')
parser.add_argument('--lgb-s', dest='lgb_subsample_frac', type=float, default=1.0, help='LGB hyperparameter.')
parser.add_argument('--lgb-cv', dest='lgb_cv', action='store_true', default=1, help='Tune LGB hyperparameters.')
args = parser.parse_args()

config = prep_env()
exp_version = 'lgb'


def evaluate_scores(turb_ids):
    exp = ExpLgb(config, exp_version)
    mae_, rmse_, score_ = 0, 0, 0
    for i in turb_ids:
        exp.train_single_turbine(i)
        mae, rmse, score = exp.test(i, test=1)

        mae_ += mae
        rmse_ += rmse
        score_ += score

    return mae_, rmse_, score_


if __name__ == '__main__':
    turbines = range(1, (config['capacity'] + 1))

    njobs = 4
    nturbs = config['capacity'] // njobs + 1
    jobload = [turbines[i:i+nturbs] for i in range(0, config['capacity'], nturbs)]
    futures = []
    maes, rmses, scores = 0., 0., 0.

    executor = ProcessPoolExecutor(max_workers=njobs)
    for idx in range(njobs):
        futures.append(executor.submit(partial(evaluate_scores, jobload[idx])))
    for future in futures:
        if future is not None:
            maes += future.result()[0]
            rmses += future.result()[1]
            scores += future.result()[2]

    with open("result.txt", 'a') as f:
        f.write(f"{config['model_type']}: Turbine all\n")
        f.write('rmse:{}, mae:{}, score:{}'.format(
            np.mean(rmses),
            np.mean(maes),
            np.mean(scores)))
        f.write('\n')
        f.write('\n')


# if __name__ == '__main__':
#     config = prep_env()
#     experiment = ExpLgb(config)
#     maes = []
#     rmses = []
#     scores = []
#     for turb_id in range(58, 69):
#         print(f'Now training for turbine {turb_id}:')
#         # experiment.train_single_turbine(turb_id)
#         mae, rmse, score = experiment.test(turb_id, test=1)
#         maes += [mae]
#         rmses += [rmse]
#         scores += [score]
#
#     f = open("result.txt", 'a')
#     f.write(f"{config['model_type']}: Turbine all\n")
#     f.write('rmse:{}, mae:{}, score:{}'.format(
#         np.mean(rmses),
#         np.mean(maes),
#         np.mean(scores)))
#     f.write('\n')
#     f.write('\n')
#     f.close()

