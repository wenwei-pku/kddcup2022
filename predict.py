# -*-Encoding: utf-8 -*-

import numpy as np
from prepare import prep_env
from pathlib import Path
from data_provider.test_data_loader import LoadDataTree
import pandas as pd
import joblib
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")



def predict_single_turbine(config, test_data, turb_id):

    model_path = Path(config['checkpoints'], config['model_type'], str(turb_id))
    if config['model_type'] == 'lgb':
        ypred = []
        sample_per_group = int(config['output_len'] / config['n_group'])
        for i in range(config['n_group']):
            cur_model = joblib.load(model_path / f'model_{i}.pkl')
            ypred.append(cur_model.predict(test_data[i]).reshape(sample_per_group, -1))
        ypred = np.concatenate(ypred, axis=0)
        return ypred



def forecast(config):

    test_data_obj = LoadDataTree(config)
    prediction = []
    for turb_id in range(1, (config['capacity']+1)):
        print('predict for turbine %s' % turb_id)
        test_list = test_data_obj.get_single_turbine_test_data(turb_id)
        pred = predict_single_turbine(config, test_list, turb_id)
        prediction.append(pred)
    return np.array(prediction)


if __name__ == '__main__':
    config = prep_env()
    prediction = forecast(config)
