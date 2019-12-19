import numpy as np
import pandas as pd

from base import Feature, get_arguments, generate_features
from feature_utils import target_encoding

import warnings
warnings.filterwarnings('ignore')


class target_encoding_by_building_id_folds1_meter1(Feature):
    def create_features(self):
        tr_feat, te_feat = target_encoding(train, test, 'meter_reading', 'building_id', folds1)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


class target_encoding_by_site_id_folds1_meter1(Feature):
    def create_features(self):
        tr_feat, te_feat = target_encoding(train, test, 'meter_reading', 'site_id', folds1)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


class target_encoding_by_primary_use_folds1_meter1(Feature):
    def create_features(self):
        tr_feat, te_feat = target_encoding(train, test, 'meter_reading', 'primary_use', folds1)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


class target_encoding_by_building_id_folds2_log_meter1(Feature):
    def create_features(self):
        tr_feat, te_feat = target_encoding(train, test, 'meter_reading', 'building_id', folds2)
        self.train[self.__class__.__name__] = np.log1p(tr_feat)
        self.test[self.__class__.__name__] = np.log1p(te_feat)


class target_encoding_by_site_id_folds2_log_meter1(Feature):
    def create_features(self):
        tr_feat, te_feat = target_encoding(train, test, 'meter_reading', 'site_id', folds2)
        self.train[self.__class__.__name__] = np.log1p(tr_feat)
        self.test[self.__class__.__name__] = np.log1p(te_feat)


class target_encoding_by_primary_use_folds2_log_meter1(Feature):
    def create_features(self):
        tr_feat, te_feat = target_encoding(train, test, 'meter_reading', 'primary_use', folds2)
        self.train[self.__class__.__name__] = np.log1p(tr_feat)
        self.test[self.__class__.__name__] = np.log1p(te_feat)


class target_encoding_last_month_meter1(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole['month'] = whole['date'].apply(lambda x: int(x[5:7]))
        df_group = whole.iloc[:len_train].groupby(['building_id', 'month'])['meter_reading'].mean().reset_index()
        df_group['month'] = df_group['month'] + 1
        idx = df_group[df_group['month'] == 13].index
        df_group.loc[idx, 'month'] = 1
        df_group.rename(columns={'meter_reading': 'mean_meter_reading_last_month'}, inplace=True)

        whole = pd.merge(whole, df_group, on=['building_id', 'month'], how='left')

        self.train[self.__class__.__name__] = whole['mean_meter_reading_last_month'].values[:len_train]
        self.test[self.__class__.__name__] = whole['mean_meter_reading_last_month'].values[len_train:]


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('../data/input/train_meter1.feather')
    test = pd.read_feather('../data/input/test_meter1.feather')

    building_metadata = pd.read_feather('../data/input/building_metadata.feather')

    train = pd.merge(train, building_metadata, on=['building_id'], how='left')
    test = pd.merge(test, building_metadata, on=['building_id'], how='left')

    len_train = len(train)

    folds1 = pd.read_feather('../folds/01_kfold_meter1.feather')
    folds2 = pd.read_feather('../folds/02_gkfold_meter1.feather')

    generate_features(globals(), args.force)