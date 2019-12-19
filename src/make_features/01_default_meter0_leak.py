import pandas as pd

from base import Feature, get_arguments, generate_features

import warnings
warnings.filterwarnings('ignore')


class building_id_meter0_leak(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train['building_id']
        self.test[self.__class__.__name__] = test['building_id']


class meter_meter0_leak(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train['meter']
        self.test[self.__class__.__name__] = test['meter']


class timestamp_meter0_leak(Feature):
    def create_features(self):
        train_date = pd.to_datetime(train['date'], format='%Y-%m-%d')
        test_date = pd.to_datetime(test['date'], format='%Y-%m-%d')

        self.train['timestamp_month_meter0_leak'] = train_date.apply(lambda x: x.month)
        self.test['timestamp_month_meter0_leak'] = test_date.apply(lambda x: x.month)

        self.train['timestamp_day_meter0_leak'] = train_date.apply(lambda x: x.day)
        self.test['timestamp_day_meter0_leak'] = test_date.apply(lambda x: x.day)

        self.train['timestamp_weekday_meter0_leak'] = train_date.apply(lambda x: x.weekday())
        self.test['timestamp_weekday_meter0_leak'] = test_date.apply(lambda x: x.weekday())

        self.train['timestamp_hour_meter0_leak'] = train_date.apply(lambda x: x.hour)
        self.test['timestamp_hour_meter0_leak'] = test_date.apply(lambda x: x.hour)


class site_id_meter0_leak(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train['site_id']
        self.test[self.__class__.__name__] = test['site_id']


class primary_use_meter0_leak(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        le = {k: i for i, k in enumerate(whole['primary_use'].unique())}
        self.train[self.__class__.__name__] = train['primary_use'].map(le)
        self.test[self.__class__.__name__] = test['primary_use'].map(le)


class site_id_primary_use_meter0_leak(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['site_id'].astype(str) + '___' + whole['primary_use'].astype(str)
        le = {k: i for i, k in enumerate(whole[self.__class__.__name__].unique())}
        self.train[self.__class__.__name__] = (train['site_id'].astype(str) + '___' + train['primary_use'].astype(str)).map(le)
        self.test[self.__class__.__name__] = (test['site_id'].astype(str) + '___' + test['primary_use'].astype(str)).map(le)


class square_feet_meter0_leak(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train['square_feet']
        self.test[self.__class__.__name__] = test['square_feet']


class year_built_meter0_leak(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train['year_built']
        self.test[self.__class__.__name__] = test['year_built']


class floor_count_meter0_leak(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train['floor_count']
        self.test[self.__class__.__name__] = test['floor_count']


class square_feet_div_floor_count_meter0_leak(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train['square_feet'] / train['floor_count']
        self.test[self.__class__.__name__] = test['square_feet'] / test['floor_count']


class square_feet_div_mean_by_site_id_meter0_leak(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['square_feet'] / whole.groupby(['site_id'])['square_feet'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class square_feet_div_mean_by_primary_use_meter0_leak(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['square_feet'] / whole.groupby(['primary_use'])['square_feet'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class square_feet_div_mean_by_year_built_meter0_leak(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['square_feet'] / whole.groupby(['year_built'])['square_feet'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('../data/input/train_and_leak_meter0.feather')
    test = pd.read_feather('../data/input/test_meter0.feather')

    building_metadata = pd.read_feather('../data/input/building_metadata.feather')

    train = pd.merge(train, building_metadata, on=['building_id'], how='left')
    test = pd.merge(test, building_metadata, on=['building_id'], how='left')

    len_train = len(train)

    generate_features(globals(), args.force)