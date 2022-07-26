"""
Feature Engineering part. For tree based models.
Adjusted from general-eforecaster toolkit.

"""

import numpy as np
from typing import List, Dict
import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import List, Union
import pandas as pd


class FeaturizerBase(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def copy(self):
        return deepcopy(self)


class WinOffsetFeaturizerBase(FeaturizerBase, metaclass=ABCMeta):

    def __init__(self,
                 wins: Union[int, List[int]] = None,
                 offsets: Union[int, List[int]] = None,
                 feature_col: List[str] = None):
        """
        Window offset featurizer base, the child class should have at least one of the following attributes:
        1. wins
        2. offsets

        Args:
            wins: list of window sizes
            offsets: list of offsets or lags
            feature_col: list of feature columns
        """
        super().__init__()

        self.wins = wins
        self.offsets = offsets
        self.feature_col = feature_col

    @property
    def wins(self):
        return self._wins

    @wins.setter
    def wins(self, value):
        if value is None:
            self._wins = []
        elif isinstance(value, int):
            self._wins = [value]
        else:
            self._wins = list(value)

    @property
    def offsets(self):
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        if value is None:
            self._offsets = []
        elif isinstance(value, int):
            self._offsets = [value]
        else:
            self._offsets = list(value)

    @property
    def feature_col(self):
        return self._feature_col

    @feature_col.setter
    def feature_col(self, value):
        if not value:
            raise ValueError('feature_col can not be empty!')
        elif isinstance(value, str):
            self._feature_col = [value]
        else:
            self._feature_col = list(value)


class DifferenceFeaturizer(WinOffsetFeaturizerBase):

    def __init__(self, *, offsets, feature_col, freq=None):
        """
        Building difference features.

        Args:
            offsets: list of lags, each element in the list corresponds to one lag.
            feature_col: list of input raw features. These features will be used to calculate stats and lag features.
            freq: The shift frequency.
        """
        super().__init__(offsets=offsets, feature_col=feature_col)

        self.freq = freq

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform df to have difference features

        Args:
            df: input df

        Returns: output df with difference features.

        """
        feature_df = df[self.feature_col]
        df_lst = [df]

        for offset in self.offsets:
            diff_df = feature_df.shift(offset, self.freq) - feature_df.shift(offset+1, self.freq)
            diff_df.columns = [
                '{}_diff_offset_{}'.format(col, offset)
                for col in feature_df.columns.tolist()
            ]
            df_lst.append(diff_df)

        return pd.concat(df_lst, axis=1)


class LagFeaturizer(WinOffsetFeaturizerBase):

    def __init__(self, *, offsets, feature_col, freq=None):
        """
        Building lag features.

        Args:
            offsets: list of lags, each element in the list corresponds to one lag.
            feature_col: list of input raw features. These features will be used to calculate stats and lag  features.
            freq: The shift frequency.
        """
        super().__init__(offsets=offsets, feature_col=feature_col)

        self.freq = freq

    #     @log_status()
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform df to have difference features

        Args:
            df: input df

        Returns: output df with difference features.

        """
        feature_df = df[self.feature_col]
        df_lst = [df]

        for offset in self.offsets:
            lag_df = feature_df.shift(offset, self.freq)
            lag_df.columns = [
                '{}_lag_offset_{}'.format(col, offset)
                for col in feature_df.columns.tolist()
            ]
            df_lst.append(lag_df)

        return pd.concat(df_lst, axis=1)


class CubeFeaturizer(WinOffsetFeaturizerBase):

    def __init__(self, *, offsets, feature_col, freq=None):
        """
        Building cubic features.

        Args:
            offsets: list of lags, each element in the list corresponds to one lag.
            feature_col: list of input raw features. These features will be used to calculate stats and lag features.
            freq: The shift frequency.
        """
        super().__init__(offsets=offsets, feature_col=feature_col)

        self.freq = freq

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform df to have difference features

        Args:
            df: input df

        Returns: output df with difference features.

        """
        feature_df = df[self.feature_col]
        df_lst = [df]

        diff_df = feature_df**3
        diff_df.columns = [
            '{}_cube'.format(col)
            for col in feature_df.columns.tolist()
        ]
        df_lst.append(diff_df)

        return pd.concat(df_lst, axis=1)

    

class RollingStatsFeaturizer(WinOffsetFeaturizerBase):
    DEFAULT_STATS = ['min', 'max', 'median', 'mean', 'std', 'skew']
    DEFAULT_ROLLING_KWARGS = {'min_periods': 1, 'center': False}

    def __init__(self,
                 *,
                 wins,
                 offsets,
                 feature_col,
                 is_interval: bool = False,
                 interval_key: str = 'index_10min',
                 freq: int = None,
                 stats: List[str] = None,
                 quantiles: List[float] = None,
                 rolling_kwargs: Dict = None):

        super().__init__(wins=wins, offsets=offsets, feature_col=feature_col)

        self.freq = freq
        self.stats = stats
        self.offsets = offsets
        self.is_interval = is_interval
        self.interval_key = interval_key
        self.quantiles = quantiles
        self.rolling_kwargs = rolling_kwargs
        self.wins = wins

        # print(self.wins)


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform df to have rolling stats features

        Args:
            df: input df

        Returns: output df with rolling stats features.

        """
        drop_interval_key = False

        if self.is_interval:
            if self.interval_key not in list(df.columns):
                print('interval key {} not in df columns, '
                      'using default index_15\0min as interval key...'.format(self.interval_key))

                self.interval_key = 'index_10min'
                index_10min = df.index.hour * 6 + df.index.minute // 10
                df = df.assign(index_10min=index_10min)
                drop_interval_key = True  # drop index_15min after transformation

            # if self.interval_key not in self.feature_col:
            #     self.feature_col.append(self.interval_key)
            # else:
            #     logging.warning('interval_key {} in feature_col, '
            #                     'this feature will not be used to'
            #                     ' calculate the interval features ...'.format(self.interval_key))

        feature_df = df[self.feature_col]
        df_lst = [df]



        for win in self.wins:
            if win <= 1:
                logging.warning('window size <= 1, this will build lags only ...')
                stats = ['mean']
                win = 1
            elif win < 3 and 'skew' in self.stats:
                logging.warning('window size {} < 3, '
                                'skew requires at least 3 samples, '
                                'it will not be calculated !'.format(win))
                stats = self.stats[:]
                stats.remove('skew')
            else:
                stats = self.stats

            for offset in self.offsets:
                if self.is_interval:
                    curr_df = self._build_interval_stats(feature_df, win, offset, stats)
                else:
                    curr_df = self._build_rolling_stats(feature_df, win, offset, stats)

                df_lst.append(curr_df)

        out_df = pd.concat(df_lst, axis=1)
        if drop_interval_key:
            out_df.drop(columns=self.interval_key, inplace=True)

        return out_df

    def _build_interval_stats(self, df, win, offset, stats):
        df_lst = []
        df_groupby = df.groupby(self.interval_key)
        for col in df.columns.tolist():
            if col != self.interval_key:
                agg_dict = {
                    f'{self.interval_key}_{col}_win_{win}_offset_{offset}_{stat}': stat
                    for stat in stats
                }
                stats_df = df_groupby[col].apply(lambda x: x.rolling(win, **self.rolling_kwargs).agg(agg_dict))
                df_lst.append(stats_df)

                if self.quantiles is not None:
                    for qtl in self.quantiles:
                        pct_df = df_groupby[col].apply(lambda x: x.rolling(win, **self.rolling_kwargs).quantile(qtl))
                        pct_df.name = f'{self.interval_key}_{col}_win_{win}_offset_{offset}_q{qtl}'
                        df_lst.append(pct_df)

        return pd.concat(df_lst, axis=1).shift(offset, self.freq)

    def _build_rolling_stats(self, df, win, offset, stats):
        df_lst = []
        for col in df.columns.tolist():
            if col != self.interval_key:
                agg_dict = {
                    f'{col}_win_{win}_offset_{offset}_{stat}': stat
                    for stat in stats
                }
                stats_df = df[col].rolling(win, **self.rolling_kwargs).agg(agg_dict)
                df_lst.append(stats_df)

                if self.quantiles is not None:
                    for qtl in self.quantiles:
                        pct_df = df[col].rolling(win, **self.rolling_kwargs).quantile(qtl)
                        pct_df.name = f'{col}_win_{win}_offset_{offset}_q{qtl}'
                        df_lst.append(pct_df)

        return pd.concat(df_lst, axis=1).shift(offset, self.freq)

    @property
    def stats(self):
        return self._stats

    @stats.setter
    def stats(self, value):
        if value is None:
            logging.warning('using all default stats {}'.format(self.DEFAULT_STATS))
            self._stats = self.DEFAULT_STATS
        else:
            self._stats = list(value)

    @property
    def rolling_kwargs(self):
        return self._rolling_kwargs

    @rolling_kwargs.setter
    def rolling_kwargs(self, value):
        if value is None:
            self._rolling_kwargs = self.DEFAULT_ROLLING_KWARGS
        elif not isinstance(value, dict):
            raise ValueError('rolling_kwargs should be a dict!')
        else:
            self._rolling_kwargs = value

    @property
    def quantiles(self):
        return self._quantiles

    @quantiles.setter
    def quantiles(self, value):
        if value is not None:
            if isinstance(value, (float, int)):
                self._quantiles = [value]
            else:
                self._quantiles = list(value)

            for qtl in self._quantiles:
                if qtl > 1 or qtl < 0:
                    raise ValueError('Percentiles have to be float between 0 and 1!')

        else:
            self._quantiles = value


class FeatureEnsemblerBase(metaclass=ABCMeta):

    def __init__(self, featurizers: List[FeaturizerBase]):
        """
        Ensemble featurizers. Featurizers should be a list of instance of FeaturizerBase

        Args:
            featurizers: List of instance of FeaturizerBase
        """
        self.featurizers = featurizers

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def featurizers(self):
        return self._featurizers

    @featurizers.setter
    def featurizers(self, value):
        if not isinstance(value, Sequence):
            value = [value]
        else:
            value = list(value)

        for featurizer in value:
            if not isinstance(featurizer, FeaturizerBase):
                raise TypeError('{} should be a instance of FeaturizerBase'.format(featurizer))

        self._featurizers = value

    def copy(self):
        return deepcopy(self)


class FeatureEnsembler(FeatureEnsemblerBase):
    def __init__(self,
                 featurizers,
                 fillna: str = 'linear',
                 label_col: str = 'load',
                 keep_filled_label: bool = False):
        """
        Ensemble features for bus load tasks.

        Args:
            featurizers: List of featurizer instances.
            fillna: df.interpolate(method=fillna)
            label_col: label column
            keep_filled_label: if False, will keep the original label column, otherwise, will
                               perform fillna for label column.
        """
        super().__init__(featurizers)
        self.fillna = fillna
        self.label_col = label_col
        self.keep_filled_label = keep_filled_label

#     @log_status()
    def transform(self, df: pd.DataFrame):
        """
        Transform df to df with various features defined by list of featurizers.

        Caution:
        1. Please ensure that input df has datetime index.

        Args:
            df: input df

        Returns: df with features ready for modeling

        """
#         if not isinstance(df.index, pd.DatetimeIndex):
#             logging.warning('df index is not type of pd.DatetimeIndex, '
#                             'this may lead to unexpected behavior in featurizers!')

        df = df.copy()
        label = df[self.label_col].copy()
        fill_cols = df.select_dtypes(include=[np.number]).columns
        df[fill_cols] = df[fill_cols].interpolate(method=self.fillna)

        logging.info('filling na for columns {} ...'.format(fill_cols))

        for featurizer in self.featurizers:
            df = featurizer.transform(df)

        df[fill_cols] = df[fill_cols].interpolate(method=self.fillna)

        # recover original label
        if not self.keep_filled_label:
            df[self.label_col] = label

        # replace symbols in column names
        # df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x), inplace=True)

        return df
