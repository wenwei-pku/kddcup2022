import torch
from prepare import prep_env
from exp.exp_lgb_dense import ExpLgb_dense
from exp.exp_lgb import ExpLgb
from datetime import datetime
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    config = prep_env()
    torch.manual_seed(config['seed'])
    exp_version = datetime.now().strftime("%Y%m%d%H%M%S")

    if config['model_type'] == 'lgb':
        exp_version = 'lgb'
        experiment = ExpLgb(config, exp_version)
    else:
        exp_version = 'lgb_dense'
        experiment = ExpLgb_dense(config, exp_version)

    maes = []
    rmses = []
    scores = []
    for turb_id in range(1, (config['capacity'] + 1)):
        print(f'Now training for turbine {turb_id}:')
        experiment.train_single_turbine(turb_id)
        mae, rmse, score = experiment.test(turb_id, test=1)

        maes += [mae]
        rmses += [rmse]
        scores += [score]

    with open("result.txt", 'a') as f:
        f.write(f"{config['model_type']}: Turbine all\n")
        f.write('rmse:{}, mae:{}, score:{}'.format(
            np.mean(rmses),
            np.mean(maes),
            np.mean(scores)))
        f.write('\n')
        f.write('\n')
