import pandas as pd
import numpy as np
from pathlib import Path
from layers.featurizer import CubeFeaturizer, DifferenceFeaturizer, LagFeaturizer, RollingStatsFeaturizer, FeatureEnsembler
import warnings
from utils.tools import iqr_outliers_mask, abnormal_mask
import joblib
warnings.filterwarnings('ignore')


class LoadDataTree():

    """
    Data loader for tree based models
    """

    def __init__(self, config):
        self.config = config
        self.data = pd.read_csv(config['path_to_test_x'], )

    def get_single_turbine(self, turb_id):

        cur_data = self.data[self.data['TurbID'] == turb_id]
        cur_data['hour_sin'] = cur_data['Tmstamp'].str.split(':').apply(lambda x: np.sin(int(x[0])))
        cur_data['hour_cos'] = cur_data['Tmstamp'].str.split(':').apply(lambda x: np.cos(int(x[0])))

        ## Deal with outliers
        cur_data['Patv'].clip(lower=0, inplace=True)

        # 1. IQR outliers removal, replaced with nan and interpolate then.
        for label in ['Etmp', 'Wspd']:
            iqr_mask = iqr_outliers_mask(cur_data[label])
            cur_data[label].mask(iqr_mask, inplace=True)
            cur_data[label].interpolate(inplace=True)

        # Mask abnormal values and missing values on train and valid sets
        caveat_mask = abnormal_mask(cur_data)
        cur_data.mask(caveat_mask, inplace=True)

        caveat_mask = abnormal_mask(cur_data)
        cur_data.mask(caveat_mask, inplace=True)

        return cur_data.iloc[-self.config['input_len']:, :]

    def get_single_turbine_test_data(self, turb_id):
        test_data = self.get_single_turbine(turb_id)
        test_data['Wspd_cube'] = test_data['Wspd'] ** 3

        test_data = self._feature_engineering_().transform(test_data[['Etmp', 'Wspd', 'Wspd_cube', 'Patv', 'hour_sin', 'hour_cos']])

        test_list = self._make_test_data_(test_data)
        return test_list

    def _feature_engineering_(self):
        difference_featurizer_params = {
            'offsets': np.r_[1:24:1],
            'feature_col': ['Wspd', 'Etmp', 'Patv']
        }

        rolling_featurizer_params = {'feature_col': ['Etmp', 'Wspd', 'Patv', 'Wspd_cube'],
                                     'wins': [3, 6, 36, 144],
                                     'quantiles': [0.25, 0.75],
                                     'offsets': 1,
                                     'stats': None,
                                     'is_interval': False,
                                     # 'rolling_kwargs': {'min_periods': 6}
                                     }

        lag_featurizer_params = {
            'offsets': np.r_[1:24:1],
            'feature_col': ['Wspd', 'Wspd_cube', 'Etmp', 'Patv']
        }

        featurizer = FeatureEnsembler([DifferenceFeaturizer(**difference_featurizer_params),
                                       LagFeaturizer(**lag_featurizer_params),
                                       RollingStatsFeaturizer(**rolling_featurizer_params),
                                       ],
                                      label_col='Patv'
                                      )

        return featurizer


    def _make_test_data_(self, test_data):
        samples_per_group = int(self.config['output_len'] / self.config['n_group'])
        test_list = []

        for i in range(self.config['n_group']):
            X = test_data.drop(['Patv', 'Etmp', 'Wspd', 'Wspd_cube'], axis=1).values[-1].reshape(1, -1)
            cur_X = np.repeat(X, repeats=samples_per_group, axis=0)
            time_index = np.repeat([np.r_[0:samples_per_group]], repeats=X.shape[0], axis=0).flatten()
            cur_X = np.concatenate((cur_X, time_index.reshape(-1, 1)), axis=1)
            test_list.append(cur_X)
        return test_list
