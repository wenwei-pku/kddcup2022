import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
from utils.tools import iqr_outliers_mask, abnormal_mask

warnings.filterwarnings('ignore')

class DatasetTree():
    """
    DataLoader for Tree based models.
    """

    def __init__(self,
                 data_path,
                 start_col=3,
                 farm_capacity=134,  ## 134 wind turbines in total
                 target='Patv',
                 day_len=144,
                 train_days=200,
                 val_days=30,
                 test_days=15,
                 input_len=144,
                 pred_len=288
                 ):
        """

        Args:
            data_path: str of Path obj.
            start_col:
            farm_capacity: # of totoal turbines on site
            target:
            day_len: 10 minutely, 144 points in total in one day
            turb_id: Turbine ID
            train_days: # of days for training.
            val_days: # of days for validating
            test_days: # of days for testing
            input_len: # input seq len
            pred_len: # pred seq len

        Returns:

         """

        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.start_col = start_col
        self.farm_capacity = farm_capacity
        self.target = target
        self.unit_size = day_len
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.test_size = test_days * self.unit_size
        self.input_len = input_len
        self.pred_len = pred_len

        # 1. read raw data and get some basic summary statistics
        self.data_raw = pd.read_csv(self.data_path, )
        self.total_days = self.data_raw['Day'].max()
        self.total_size = self.total_days * self.unit_size


    def get_single_turbine(self, turb_id):
        """
        Get raw data from a single turbine
        Args:
            turb_id: int

        Returns:

        """
        cur_data = self.data_raw[self.data_raw['TurbID'] == turb_id].copy()
        cur_data['hour_sin'] = cur_data['Tmstamp'].str.split(':').apply(lambda x: np.sin(int(x[0])))
        cur_data['hour_cos'] = cur_data['Tmstamp'].str.split(':').apply(lambda x: np.cos(int(x[0])))
        cur_data = cur_data.reset_index()
        cur_data['index'] %= self.total_size
        cur_data = cur_data.set_index('index')
        return cur_data


    def train_test_split(self, turb_id):

        cur_data_raw = self.get_single_turbine(turb_id)
        cur_data = cur_data_raw.copy(deep=True)


        ## Deal with outliers
        cur_data['Patv'].clip(lower=0, inplace=True)

        # 1. IQR outliers removal, replaced with nan and interpolate then.
        for label in ['Etmp', 'Wspd']:
            iqr_mask = iqr_outliers_mask(cur_data[label])
            cur_data[label].mask(iqr_mask, inplace=True)
            cur_data[label].interpolate(inplace=True)


        border_ref = {
            'train': [0, self.train_size],
            'valid': [self.train_size - self.input_len, self.train_size + self.val_size],
            'test': [self.train_size + self.val_size - self.input_len,
                     self.train_size + self.val_size + self.test_size],
        }

        train = cur_data[border_ref['train'][0]: border_ref['train'][1]]
        val = cur_data[border_ref['valid'][0]: border_ref['valid'][1]]
        test = cur_data[border_ref['test'][0]: border_ref['test'][1]]

        # Mask abnormal values and missing values on train and valid sets
        caveat_mask = abnormal_mask(train)
        train.mask(caveat_mask, inplace=True)

        caveat_mask = abnormal_mask(val)
        val.mask(caveat_mask, inplace=True)

        return train, val, test, cur_data_raw[border_ref['test'][0]: border_ref['test'][1]]


if __name__ == '__main__':
    obj = DatasetTree(data_path='../datasets/wtbdata_245days.csv')
    train, val, test, cur_data_raw = obj.train_test_split(turb_id=1)
