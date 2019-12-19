import datetime
import pandas as pd

from base import Feature, get_arguments, generate_features

import warnings
warnings.filterwarnings('ignore')


class air_temperature_meter2_leak(Feature):
    def create_features(self):
        aggs = {'min', 'max', 'std', 'mean'}
        rename = {agg: agg + '_' + self.__class__.__name__ for agg in aggs}
        df_groupby = weather.groupby(['site_id', 'date'])['air_temperature'].agg(aggs).reset_index()
        df_groupby.rename(columns=rename, inplace=True)

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        for col in rename.values():
            self.train[col] = whole[col].values[:len_train]
            self.test[col] = whole[col].values[len_train:]


class diff_min_air_temperature_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['air_temperature'].min().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'air_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'air_temperature']]
        df_groupby2.rename(columns={'air_temperature': 'air_temperature_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['air_temperature'] - df_groupby['air_temperature_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_min_air_temperature_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['air_temperature'].min().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'air_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'air_temperature']]
        df_groupby2.rename(columns={'air_temperature': 'air_temperature_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['air_temperature'] - df_groupby['air_temperature_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_air_temperature_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['air_temperature'].max().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'air_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'air_temperature']]
        df_groupby2.rename(columns={'air_temperature': 'air_temperature_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['air_temperature'] - df_groupby['air_temperature_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_air_temperature_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['air_temperature'].max().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'air_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'air_temperature']]
        df_groupby2.rename(columns={'air_temperature': 'air_temperature_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['air_temperature'] - df_groupby['air_temperature_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_air_temperature_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['air_temperature'].std().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'air_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'air_temperature']]
        df_groupby2.rename(columns={'air_temperature': 'air_temperature_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['air_temperature'] - df_groupby['air_temperature_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_air_temperature_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['air_temperature'].std().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'air_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'air_temperature']]
        df_groupby2.rename(columns={'air_temperature': 'air_temperature_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['air_temperature'] - df_groupby['air_temperature_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_air_temperature_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['air_temperature'].mean().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'air_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'air_temperature']]
        df_groupby2.rename(columns={'air_temperature': 'air_temperature_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['air_temperature'] - df_groupby['air_temperature_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_air_temperature_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['air_temperature'].mean().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'air_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'air_temperature']]
        df_groupby2.rename(columns={'air_temperature': 'air_temperature_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['air_temperature'] - df_groupby['air_temperature_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class cloud_coverage_meter2_leak(Feature):
    def create_features(self):
        aggs = {'min', 'max', 'std', 'mean'}
        rename = {agg: agg + '_' + self.__class__.__name__ for agg in aggs}
        df_groupby = weather.groupby(['site_id', 'date'])['cloud_coverage'].agg(aggs).reset_index()
        df_groupby.rename(columns=rename, inplace=True)

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        for col in rename.values():
            self.train[col] = whole[col].values[:len_train]
            self.test[col] = whole[col].values[len_train:]


class diff_min_cloud_coverage_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['cloud_coverage'].min().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'cloud_coverage']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'cloud_coverage']]
        df_groupby2.rename(columns={'cloud_coverage': 'cloud_coverage_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['cloud_coverage'] - df_groupby['cloud_coverage_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_min_cloud_coverage_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['cloud_coverage'].min().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'cloud_coverage']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'cloud_coverage']]
        df_groupby2.rename(columns={'cloud_coverage': 'cloud_coverage_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['cloud_coverage'] - df_groupby['cloud_coverage_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_cloud_coverage_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['cloud_coverage'].max().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'cloud_coverage']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'cloud_coverage']]
        df_groupby2.rename(columns={'cloud_coverage': 'cloud_coverage_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['cloud_coverage'] - df_groupby['cloud_coverage_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_cloud_coverage_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['cloud_coverage'].max().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'cloud_coverage']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'cloud_coverage']]
        df_groupby2.rename(columns={'cloud_coverage': 'cloud_coverage_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['cloud_coverage'] - df_groupby['cloud_coverage_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_cloud_coverage_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['cloud_coverage'].std().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'cloud_coverage']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'cloud_coverage']]
        df_groupby2.rename(columns={'cloud_coverage': 'cloud_coverage_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['cloud_coverage'] - df_groupby['cloud_coverage_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_cloud_coverage_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['cloud_coverage'].std().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'cloud_coverage']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'cloud_coverage']]
        df_groupby2.rename(columns={'cloud_coverage': 'cloud_coverage_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['cloud_coverage'] - df_groupby['cloud_coverage_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_cloud_coverage_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['cloud_coverage'].mean().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'cloud_coverage']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'cloud_coverage']]
        df_groupby2.rename(columns={'cloud_coverage': 'cloud_coverage_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['cloud_coverage'] - df_groupby['cloud_coverage_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_cloud_coverage_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['cloud_coverage'].mean().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'cloud_coverage']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'cloud_coverage']]
        df_groupby2.rename(columns={'cloud_coverage': 'cloud_coverage_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['cloud_coverage'] - df_groupby['cloud_coverage_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class dew_temperature_meter2_leak(Feature):
    def create_features(self):
        aggs = {'min', 'max', 'std', 'mean'}
        rename = {agg: agg + '_' + self.__class__.__name__ for agg in aggs}
        df_groupby = weather.groupby(['site_id', 'date'])['dew_temperature'].agg(aggs).reset_index()
        df_groupby.rename(columns=rename, inplace=True)

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        for col in rename.values():
            self.train[col] = whole[col].values[:len_train]
            self.test[col] = whole[col].values[len_train:]


class diff_min_dew_temperature_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['dew_temperature'].min().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'dew_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'dew_temperature']]
        df_groupby2.rename(columns={'dew_temperature': 'dew_temperature_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['dew_temperature'] - df_groupby['dew_temperature_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_min_dew_temperature_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['dew_temperature'].min().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'dew_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'dew_temperature']]
        df_groupby2.rename(columns={'dew_temperature': 'dew_temperature_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['dew_temperature'] - df_groupby['dew_temperature_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_dew_temperature_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['dew_temperature'].max().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'dew_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'dew_temperature']]
        df_groupby2.rename(columns={'dew_temperature': 'dew_temperature_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['dew_temperature'] - df_groupby['dew_temperature_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_dew_temperature_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['dew_temperature'].max().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'dew_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'dew_temperature']]
        df_groupby2.rename(columns={'dew_temperature': 'dew_temperature_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['dew_temperature'] - df_groupby['dew_temperature_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_dew_temperature_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['dew_temperature'].std().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'dew_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'dew_temperature']]
        df_groupby2.rename(columns={'dew_temperature': 'dew_temperature_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['dew_temperature'] - df_groupby['dew_temperature_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_dew_temperature_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['dew_temperature'].std().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'dew_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'dew_temperature']]
        df_groupby2.rename(columns={'dew_temperature': 'dew_temperature_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['dew_temperature'] - df_groupby['dew_temperature_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_dew_temperature_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['dew_temperature'].mean().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'dew_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'dew_temperature']]
        df_groupby2.rename(columns={'dew_temperature': 'dew_temperature_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['dew_temperature'] - df_groupby['dew_temperature_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_dew_temperature_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['dew_temperature'].mean().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'dew_temperature']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'dew_temperature']]
        df_groupby2.rename(columns={'dew_temperature': 'dew_temperature_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['dew_temperature'] - df_groupby['dew_temperature_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class precip_depth_1_hr_meter2_leak(Feature):
    def create_features(self):
        aggs = {'min', 'max', 'std', 'mean'}
        rename = {agg: agg + '_' + self.__class__.__name__ for agg in aggs}
        df_groupby = weather.groupby(['site_id', 'date'])['precip_depth_1_hr'].agg(aggs).reset_index()
        df_groupby.rename(columns=rename, inplace=True)

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        for col in rename.values():
            self.train[col] = whole[col].values[:len_train]
            self.test[col] = whole[col].values[len_train:]


class diff_min_precip_depth_1_hr_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['precip_depth_1_hr'].min().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'precip_depth_1_hr']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'precip_depth_1_hr']]
        df_groupby2.rename(columns={'precip_depth_1_hr': 'precip_depth_1_hr_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['precip_depth_1_hr'] - df_groupby['precip_depth_1_hr_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_min_precip_depth_1_hr_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['precip_depth_1_hr'].min().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'precip_depth_1_hr']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'precip_depth_1_hr']]
        df_groupby2.rename(columns={'precip_depth_1_hr': 'precip_depth_1_hr_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['precip_depth_1_hr'] - df_groupby['precip_depth_1_hr_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_precip_depth_1_hr_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['precip_depth_1_hr'].max().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'precip_depth_1_hr']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'precip_depth_1_hr']]
        df_groupby2.rename(columns={'precip_depth_1_hr': 'precip_depth_1_hr_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['precip_depth_1_hr'] - df_groupby['precip_depth_1_hr_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_precip_depth_1_hr_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['precip_depth_1_hr'].max().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'precip_depth_1_hr']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'precip_depth_1_hr']]
        df_groupby2.rename(columns={'precip_depth_1_hr': 'precip_depth_1_hr_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['precip_depth_1_hr'] - df_groupby['precip_depth_1_hr_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_precip_depth_1_hr_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['precip_depth_1_hr'].std().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'precip_depth_1_hr']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'precip_depth_1_hr']]
        df_groupby2.rename(columns={'precip_depth_1_hr': 'precip_depth_1_hr_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['precip_depth_1_hr'] - df_groupby['precip_depth_1_hr_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_precip_depth_1_hr_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['precip_depth_1_hr'].std().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'precip_depth_1_hr']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'precip_depth_1_hr']]
        df_groupby2.rename(columns={'precip_depth_1_hr': 'precip_depth_1_hr_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['precip_depth_1_hr'] - df_groupby['precip_depth_1_hr_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_precip_depth_1_hr_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['precip_depth_1_hr'].mean().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'precip_depth_1_hr']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'precip_depth_1_hr']]
        df_groupby2.rename(columns={'precip_depth_1_hr': 'precip_depth_1_hr_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['precip_depth_1_hr'] - df_groupby['precip_depth_1_hr_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_precip_depth_1_hr_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['precip_depth_1_hr'].mean().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'precip_depth_1_hr']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'precip_depth_1_hr']]
        df_groupby2.rename(columns={'precip_depth_1_hr': 'precip_depth_1_hr_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['precip_depth_1_hr'] - df_groupby['precip_depth_1_hr_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class sea_level_pressure_meter2_leak(Feature):
    def create_features(self):
        aggs = {'min', 'max', 'std', 'mean'}
        rename = {agg: agg + '_' + self.__class__.__name__ for agg in aggs}
        df_groupby = weather.groupby(['site_id', 'date'])['sea_level_pressure'].agg(aggs).reset_index()
        df_groupby.rename(columns=rename, inplace=True)

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        for col in rename.values():
            self.train[col] = whole[col].values[:len_train]
            self.test[col] = whole[col].values[len_train:]


class diff_min_sea_level_pressure_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['sea_level_pressure'].min().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'sea_level_pressure']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'sea_level_pressure']]
        df_groupby2.rename(columns={'sea_level_pressure': 'sea_level_pressure_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['sea_level_pressure'] - df_groupby['sea_level_pressure_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_min_sea_level_pressure_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['sea_level_pressure'].min().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'sea_level_pressure']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'sea_level_pressure']]
        df_groupby2.rename(columns={'sea_level_pressure': 'sea_level_pressure_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['sea_level_pressure'] - df_groupby['sea_level_pressure_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_sea_level_pressure_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['sea_level_pressure'].max().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'sea_level_pressure']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'sea_level_pressure']]
        df_groupby2.rename(columns={'sea_level_pressure': 'sea_level_pressure_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['sea_level_pressure'] - df_groupby['sea_level_pressure_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_sea_level_pressure_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['sea_level_pressure'].max().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'sea_level_pressure']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'sea_level_pressure']]
        df_groupby2.rename(columns={'sea_level_pressure': 'sea_level_pressure_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['sea_level_pressure'] - df_groupby['sea_level_pressure_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_sea_level_pressure_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['sea_level_pressure'].std().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'sea_level_pressure']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'sea_level_pressure']]
        df_groupby2.rename(columns={'sea_level_pressure': 'sea_level_pressure_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['sea_level_pressure'] - df_groupby['sea_level_pressure_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_sea_level_pressure_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['sea_level_pressure'].std().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'sea_level_pressure']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'sea_level_pressure']]
        df_groupby2.rename(columns={'sea_level_pressure': 'sea_level_pressure_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['sea_level_pressure'] - df_groupby['sea_level_pressure_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_sea_level_pressure_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['sea_level_pressure'].mean().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'sea_level_pressure']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'sea_level_pressure']]
        df_groupby2.rename(columns={'sea_level_pressure': 'sea_level_pressure_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['sea_level_pressure'] - df_groupby['sea_level_pressure_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_sea_level_pressure_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['sea_level_pressure'].mean().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'sea_level_pressure']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'sea_level_pressure']]
        df_groupby2.rename(columns={'sea_level_pressure': 'sea_level_pressure_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['sea_level_pressure'] - df_groupby['sea_level_pressure_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class wind_direction_meter2_leak(Feature):
    def create_features(self):
        aggs = {'min', 'max', 'std', 'mean'}
        rename = {agg: agg + '_' + self.__class__.__name__ for agg in aggs}
        df_groupby = weather.groupby(['site_id', 'date'])['wind_direction'].agg(aggs).reset_index()
        df_groupby.rename(columns=rename, inplace=True)

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        for col in rename.values():
            self.train[col] = whole[col].values[:len_train]
            self.test[col] = whole[col].values[len_train:]


class diff_min_wind_direction_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_direction'].min().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_direction']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'wind_direction']]
        df_groupby2.rename(columns={'wind_direction': 'wind_direction_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_direction'] - df_groupby['wind_direction_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_min_wind_direction_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_direction'].min().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_direction']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'wind_direction']]
        df_groupby2.rename(columns={'wind_direction': 'wind_direction_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_direction'] - df_groupby['wind_direction_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_wind_direction_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_direction'].max().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_direction']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'wind_direction']]
        df_groupby2.rename(columns={'wind_direction': 'wind_direction_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_direction'] - df_groupby['wind_direction_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_wind_direction_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_direction'].max().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_direction']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'wind_direction']]
        df_groupby2.rename(columns={'wind_direction': 'wind_direction_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_direction'] - df_groupby['wind_direction_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_wind_direction_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_direction'].std().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_direction']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'wind_direction']]
        df_groupby2.rename(columns={'wind_direction': 'wind_direction_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_direction'] - df_groupby['wind_direction_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_wind_direction_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_direction'].std().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_direction']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'wind_direction']]
        df_groupby2.rename(columns={'wind_direction': 'wind_direction_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_direction'] - df_groupby['wind_direction_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_wind_direction_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_direction'].mean().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_direction']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'wind_direction']]
        df_groupby2.rename(columns={'wind_direction': 'wind_direction_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_direction'] - df_groupby['wind_direction_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_wind_direction_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_direction'].mean().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_direction']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'wind_direction']]
        df_groupby2.rename(columns={'wind_direction': 'wind_direction_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_direction'] - df_groupby['wind_direction_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class wind_speed_meter2_leak(Feature):
    def create_features(self):
        aggs = {'min', 'max', 'std', 'mean'}
        rename = {agg: agg + '_' + self.__class__.__name__ for agg in aggs}
        df_groupby = weather.groupby(['site_id', 'date'])['wind_speed'].agg(aggs).reset_index()
        df_groupby.rename(columns=rename, inplace=True)

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        for col in rename.values():
            self.train[col] = whole[col].values[:len_train]
            self.test[col] = whole[col].values[len_train:]


class diff_min_wind_speed_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_speed'].min().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_speed']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'wind_speed']]
        df_groupby2.rename(columns={'wind_speed': 'wind_speed_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_speed'] - df_groupby['wind_speed_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_min_wind_speed_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_speed'].min().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_speed']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'wind_speed']]
        df_groupby2.rename(columns={'wind_speed': 'wind_speed_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_speed'] - df_groupby['wind_speed_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_wind_speed_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_speed'].max().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_speed']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'wind_speed']]
        df_groupby2.rename(columns={'wind_speed': 'wind_speed_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_speed'] - df_groupby['wind_speed_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_max_wind_speed_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_speed'].max().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_speed']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'wind_speed']]
        df_groupby2.rename(columns={'wind_speed': 'wind_speed_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_speed'] - df_groupby['wind_speed_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_wind_speed_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_speed'].std().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_speed']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'wind_speed']]
        df_groupby2.rename(columns={'wind_speed': 'wind_speed_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_speed'] - df_groupby['wind_speed_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_std_wind_speed_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_speed'].std().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_speed']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'wind_speed']]
        df_groupby2.rename(columns={'wind_speed': 'wind_speed_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_speed'] - df_groupby['wind_speed_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_wind_speed_from_pre_1d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_speed'].mean().reset_index()
        df_groupby['date_pre_1d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=1))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_speed']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_1d', 'wind_speed']]
        df_groupby2.rename(columns={'wind_speed': 'wind_speed_pre_1d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_1d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_speed'] - df_groupby['wind_speed_pre_1d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


class diff_mean_wind_speed_from_pre_7d_meter2_leak(Feature):
    def create_features(self):
        df_groupby = weather.groupby(['site_id', 'date'])['wind_speed'].mean().reset_index()
        df_groupby['date_pre_7d'] = (pd.to_datetime(df_groupby['date']).apply(lambda x: x + datetime.timedelta(days=7))).astype(str)
        df_groupby1 = df_groupby[['site_id', 'date', 'wind_speed']]
        df_groupby2 = df_groupby[['site_id', 'date_pre_7d', 'wind_speed']]
        df_groupby2.rename(columns={'wind_speed': 'wind_speed_pre_7d'}, inplace=True)

        df_groupby = pd.merge(df_groupby1, df_groupby2, left_on=['site_id', 'date'], right_on=['site_id', 'date_pre_7d'], how='left')
        df_groupby[self.__class__.__name__] = df_groupby['wind_speed'] - df_groupby['wind_speed_pre_7d']

        whole = pd.concat([train, test], axis=0)
        whole = pd.merge(whole, df_groupby, on=['site_id', 'date'], how='left')

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('../data/input/train_and_leak_meter2.feather')
    test = pd.read_feather('../data/input/test_meter2.feather')

    building_metadata = pd.read_feather('../data/input/building_metadata.feather')

    train = pd.merge(train, building_metadata, on=['building_id'], how='left')
    test = pd.merge(test, building_metadata, on=['building_id'], how='left')

    weather_train = pd.read_feather('../data/input/weather_train.feather')
    weather_test = pd.read_feather('../data/input/weather_test.feather')
    weather = pd.concat([weather_train, weather_test], axis=0)
    weather['date'] = weather['timestamp'].apply(lambda x: x[:10])

    len_train = len(train)

    generate_features(globals(), args.force)