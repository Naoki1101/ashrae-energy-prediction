import pandas as pd

from base import Feature, get_arguments, generate_features

import warnings
warnings.filterwarnings('ignore')


# ===============
# train, test
# ===============
class building_id(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class meter(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# class timestamp(Feature):
#     def create_features(self):
#         train_timestamp = pd.to_datetime(train['timestamp'])
#         test_timestamp = pd.to_datetime(test['timestamp'])

#         self.train['timestamp_month'] = train_timestamp.apply(lambda x: x.month)
#         self.test['timestamp_month'] = test_timestamp.apply(lambda x: x.month)

#         self.train['timestamp_day'] = train_timestamp.apply(lambda x: x.day)
#         self.test['timestamp_day'] = test_timestamp.apply(lambda x: x.day)

#         self.train['timestamp_weekday'] = train_timestamp.apply(lambda x: x.weekday())
#         self.test['timestamp_weekday'] = test_timestamp.apply(lambda x: x.weekday())

#         self.train['timestamp_hour'] = train_timestamp.apply(lambda x: x.hour)
#         self.test['timestamp_hour'] = test_timestamp.apply(lambda x: x.hour)


class site_id(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class primary_use(Feature):
    def create_features(self):
        all_primary_use = pd.concat([train, test], axis=0)['primary_use'].unique()
        le = {k: i for i, k in enumerate(all_primary_use)}
        self.train[self.__class__.__name__] = train['primary_use'].map(le)
        self.test[self.__class__.__name__] = test['primary_use'].map(le)


class square_feet(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class year_built(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class floor_count(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class air_temperature(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class cloud_coverage(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class dew_temperature(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class precip_depth_1_hr(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class sea_level_pressure(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class wind_direction(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


class wind_speed(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('../data/input/train_.feather')
    test = pd.read_feather('../data/input/test_.feather')

    generate_features(globals(), args.force)
