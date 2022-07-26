from models.lgb import train_LGB
from layers.featurizer import DifferenceFeaturizer, LagFeaturizer, RollingStatsFeaturizer, FeatureEnsembler
import joblib
from utils.metrics import *
from pathlib import Path
import lightgbm as lgb
from data_provider.data_loader import DatasetTree
import time


class ExpLgb():
    def __init__(self, config, exp_version=None):

        """
        Args:
            config: Dict from prep_env()
        """
        self.config = config
        self.exp_version = exp_version if exp_version is not None else self.config['model_type']
        self.data_provider = DatasetTree(
            Path(self.config['data_path'], self.config['filename']),
            train_days=self.config['train_days'],
            val_days=self.config['val_days'],
            test_days=self.config['test_days'],
            input_len=self.config['input_len'],
            pred_len=self.config['output_len']
        )
        self.samples_per_group = int(self.config['output_len'] / self.config['n_group'])

        Path(self.config['result_path']).mkdir(parents=True, exist_ok=True)
        Path(self.config['checkpoints']).mkdir(parents=True, exist_ok=True)

        self.model_path = Path(self.config['checkpoints'], self.exp_version)

    def _build_feature(self,
                       difference_featurizer_params=None,
                       lag_featurizer_params=None,
                       rolling_featurerizer_params=None,
                       ):

        if difference_featurizer_params is None:
            difference_featurizer_params = {
                'offsets': np.r_[1:24:1],
                'feature_col': ['Wspd', 'Etmp', 'Patv']
            }

        if lag_featurizer_params is None:
            lag_featurizer_params = {
                'offsets': np.r_[1:24:1],
                'feature_col': ['Wspd', 'Wspd_cube', 'Etmp', 'Patv']
            }

        if rolling_featurerizer_params is None:
            rolling_featurizer_params = {'feature_col': ['Etmp', 'Wspd', 'Patv', 'Wspd_cube'],
                                           'wins': [3, 6, 36, 144],
                                           'quantiles': [0.25, 0.75],
                                           'offsets': 1,
                                           'stats': None,
                                           'is_interval': False,
                                           # 'rolling_kwargs': {'min_periods': 6}
                                           }

        featurizer = FeatureEnsembler([DifferenceFeaturizer(**difference_featurizer_params),
                                       LagFeaturizer(**lag_featurizer_params),
                                       RollingStatsFeaturizer(**rolling_featurizer_params),
                                       ],
                                      label_col='Patv'
                                      )

        return featurizer

    def _get_data(self, turb_id):
        train, val, test, test_raw = self.data_provider.train_test_split(turb_id)
        train = pd.concat([train, val, test])

        # add cube of wspd
        train['Wspd_cube'] = train['Wspd'] ** 3
        val['Wspd_cube'] = val['Wspd'] ** 3
        test['Wspd_cube'] = test['Wspd'] ** 3

        train['Wspd_u'] = train['Wspd'] * np.sin(train['Wdir'])
        train['Wspd_v'] = train['Wspd'] * np.cos(train['Wdir'])
        val['Wspd_u'] = val['Wspd'] * np.sin(val['Wdir'])
        val['Wspd_v'] = val['Wspd'] * np.cos(val['Wdir'])
        test['Wspd_u'] = test['Wspd'] * np.sin(test['Wdir'])
        test['Wspd_v'] = test['Wspd'] * np.cos(test['Wdir'])

        featurizer = self._build_feature()

        cur_checkpoint_path = self.model_path / str(turb_id)
        cur_checkpoint_path.mkdir(parents=True, exist_ok=True)

        train_data = featurizer.transform(train[['Etmp', 'Wspd', 'Wspd_cube', 'Patv', 'Wspd_u', 'Wspd_v', 'hour_sin', 'hour_cos']])
        test_data = featurizer.transform(test[['Etmp', 'Wspd', 'Wspd_cube', 'Patv', 'Wspd_u', 'Wspd_v', 'hour_sin', 'hour_cos']])
        valid_data = featurizer.transform(val[['Etmp', 'Wspd', 'Wspd_cube', 'Patv', 'Wspd_u', 'Wspd_v', 'hour_sin', 'hour_cos']])

        train_list = make_train_data(self.config['output_len'], self.config['n_group'], train_data)
        valid_list = make_train_data(self.config['output_len'], self.config['n_group'], valid_data)
        test_list, test_y = make_test_data(self.config['output_len'], self.config['n_group'], test_data)
        print('Data preprocessing ready. Start training')

        return train_list, valid_list, test_list, test_y, test_raw

    def run(self):
        res = pd.DataFrame(index=np.r_[1:(self.config['capacity'] + 1):1], columns=['mae', 'rmse', 'score'])

        for turb_id in range(1, (self.config['capacity'] + 1), 1):
            start_time = time.time()
            cur_mae, cur_rmse = self.train_single_turbine(turb_id)
            res.loc[turb_id, 'mae'] = cur_mae
            res.loc[turb_id, 'rmse'] = cur_rmse
            res.loc[turb_id, 'score'] = (res.loc[turb_id, 'mae'] + res.loc[turb_id, 'rmse']) / 2

            print(f'{(time.time() - start_time):.02f} seconds passed to train for turbine {turb_id}')

        res.to_csv(Path(self.config['result_path'], f'{self.exp_version}_result.csv'))

    def test(self, turb_id, test):
        # train, val, test, test_raw = self.data_provider.train_test_split(turb_id)
        _, _, test_list, test_y, test_raw = self._get_data(turb_id)
        cur_checkpoint_path = self.model_path / str(turb_id)
        cur_checkpoint_path.mkdir(parents=True, exist_ok=True)
        ypred = []

        for i in range(self.config['n_group']):
            cur_model = joblib.load(str(cur_checkpoint_path / f'model_{i}.pkl'))
            # print('Load model successfully.')
            cur_pred = cur_model.predict(test_list[i])
            ypred.append(cur_pred.reshape(-1, self.samples_per_group))
        ypred = np.concatenate(ypred, axis=1)

        ypred = np.expand_dims(ypred, axis=-1)
        test_y = np.expand_dims(test_y, axis=-1)

        np.save(f'results/preds_{turb_id}.npy', ypred)
        np.save(f'results/trues_{turb_id}.npy', test_y)

        maes = []
        rmses = []
        scores = []
        for j in range(self.config['input_len'], ypred.shape[0]):
            cur_mae, cur_rmse = turbine_scores(ypred[j, :, :],
                                               test_y[j, :, :],
                                               test_raw[j: j + self.config['output_len']],
                                               # examine_len=int(1e5))
                                               examine_len=self.config['output_len'])
            maes.append(cur_mae)
            rmses.append(cur_rmse)
            scores.append((cur_mae + cur_rmse) / 2)

        mae = np.mean(maes)
        rmse = np.mean(rmses)
        score = np.mean(scores)

        # print and write results
        print('rmse:{}, mae:{}, score:{}'.format(rmse, mae, score))
        f = open("result.txt", 'a')
        f.write(f"{self.config['model_type']}: Turbine {turb_id}\n")
        f.write('rmse:{}, mae:{}, score:{}'.format(rmse, mae, score))
        f.write('\n')
        f.write('\n')
        f.close()

        return mae, rmse, score

    def train_single_turbine(self, turb_id):
        train_list, valid_list, test_list, test_y, test_raw = self._get_data(turb_id)
        cur_checkpoint_path = Path(self.config['checkpoints'], self.config['model_type'], str(turb_id))
        cur_checkpoint_path.mkdir(parents=True, exist_ok=True)

        for i in range(self.config['n_group']):
            cur_X, cur_y = train_list[i]
            cur_val_X, cur_val_y = valid_list[i]

            train_set = lgb.Dataset(cur_X, cur_y, categorical_feature=[cur_X.shape[0] - 1])
            val_set = lgb.Dataset(cur_val_X, cur_val_y, categorical_feature=[cur_val_X.shape[0] - 1])

            model_lgb = train_LGB(train_set, val_set, turb_id)
            # ypred.append(model_lgb.predict(test_list[i]).reshape(-1, self.samples_per_group))
            joblib.dump(model_lgb, str(cur_checkpoint_path / f'model_{i}.pkl'))


def make_train_data(output_len, n_groups, train_data):
    samples_per_group = int(output_len / n_groups)
    train_list = []

    for i in range(n_groups):
        train_X = train_data.drop(['Patv', 'Etmp', 'Wspd', 'Wspd_cube', 'Wspd_u', 'Wspd_v'], axis=1).values[
                  :-output_len + 1]
        train_y = train_data['Patv'].shift(-(i * samples_per_group)).values

        cur_X = np.repeat(train_X, repeats=samples_per_group, axis=0)
        time_index = np.repeat([np.r_[0:samples_per_group]], repeats=train_X.shape[0], axis=0).flatten()

        cur_X = np.concatenate((cur_X, time_index.reshape(-1, 1)), axis=1)

        cur_y = []
        for j in range(train_X.shape[0]):
            cur_y.extend(train_y[j: (j + samples_per_group)])
        cur_y = np.array(cur_y)

        train_list.append((cur_X, cur_y))

    return train_list


def make_test_data(output_len, n_groups, test_data):
    samples_per_group = int(output_len / n_groups)
    test_list = []
    test_y = []

    for _ in range(n_groups):
        X = test_data.drop(['Patv', 'Etmp', 'Wspd', 'Wspd_cube', 'Wspd_u', 'Wspd_v'], axis=1).values[:-output_len + 1]
        cur_X = np.repeat(X, repeats=samples_per_group, axis=0)
        time_index = np.repeat([np.r_[0:samples_per_group]], repeats=X.shape[0], axis=0).flatten()
        cur_X = np.concatenate((cur_X, time_index.reshape(-1, 1)), axis=1)
        test_list.append(cur_X)

    for j in range(test_data.shape[0] - output_len + 1):
        cur_true_seq = test_data['Patv'].values[j: j + output_len]
        test_y.append(cur_true_seq)

    return test_list, test_y
