"""
Description: LGB baselines
"""

import lightgbm as lgb
from lightgbm import LGBMRegressor
import numpy as np


def train_LGB(train_set, valid_set=None, turb_id=1):
    fixed_param_1 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 63,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'seed': 16,
        'min_split_gain': 0.05,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_13 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_split_gain': 0.1,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_25 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'seed': 16,
        'n_estimators': 50,
        'verbosity': -1,
    }

    fixed_param_36 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 63,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_split_gain': 0.1,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_48 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_split_gain': 0.05,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_58 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 63,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_split_gain': 0.1,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_69 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_split_gain': 0.05,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_80 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_split_gain': 0.05,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_91 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_split_gain': 0.05,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_102 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 63,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_113 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_split_gain': 0.1,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    fixed_param_124 = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_split_gain': 0.05,
        'seed': 16,
        'n_estimators': 80,
        'verbosity': -1,
    }

    # if turb_id <= 12:
    #     fixed_param = fixed_param_1
    # elif turb_id <= 24:
    #     fixed_param = fixed_param_13
    # elif turb_id <= 35:
    #     fixed_param = fixed_param_25
    # elif turb_id <= 47:
    #     fixed_param = fixed_param_36
    # elif turb_id <= 57:
    #     fixed_param = fixed_param_48
    # elif turb_id <= 68:
    #     fixed_param = fixed_param_58
    # elif turb_id <= 79:
    #     fixed_param = fixed_param_69
    # elif turb_id <= 90:
    #     fixed_param = fixed_param_80
    # elif turb_id <= 101:
    #     fixed_param = fixed_param_91
    # elif turb_id <= 112:
    #     fixed_param = fixed_param_102
    # elif turb_id <= 123:
    #     fixed_param = fixed_param_113
    # else:
    #     fixed_param = fixed_param_124

    fixed_param = {
        #                 'num_threads': 12,
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'num_leaves': 40,
        'min_data_in_leaf': 60,
        'feature_fraction': 0.3,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'max_depth': 5,
        'seed': 16,
        'verbosity': -1,
        #                 'max_bins': 300,
        'num_iterations': 64,
    }

    if valid_set is not None:
        model = lgb.train(fixed_param,
                          train_set,
                          num_boost_round=20000,
                          valid_sets=[train_set],
                          early_stopping_rounds=5000,
                          verbose_eval=False,
                          )
    else:
        model = lgb.train(fixed_param,
                          train_set,
                          num_boost_round=20000,
                          valid_sets=[train_set],
                          early_stopping_rounds=5000,
                          verbose_eval=False,
                          )

    # model = LGBMRegressor(**fixed_param)
    # train_set.label[np.isnan(train_set.label)] = 0.
    #
    # model.fit(train_set.data, train_set.label)

    return model
