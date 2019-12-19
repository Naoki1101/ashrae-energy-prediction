import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train.feather')
    train['date'] = train['timestamp'].apply(lambda x: x[:10])
    train['hour'] = train['timestamp'].apply(lambda x: int(x[11:13]))

    train['timestamp'] = pd.to_datetime(train['timestamp'], format='%Y%m%d %H:%M:%S')
    train['dayofweek'] = train['timestamp'].apply(lambda x: x.weekday())


    df_sum = train.groupby(['building_id', 'meter', 'date'])['meter_reading'].sum().reset_index()
    df_sum.rename(columns={'meter_reading': 'meter_reading_day'}, inplace=True)

    train_ = pd.merge(train, df_sum, on=['building_id', 'meter', 'date'], how='left')

    s = 0.001
    train_['rate'] = train_['meter_reading'] / (train_['meter_reading_day'] + s)
    train_ = train_[train_['rate'] > 0]

    df_rate = train_.groupby(['building_id', 'meter', 'dayofweek', 'hour'])['rate'].mean().reset_index()

    df_rate.reset_index(drop=True).to_feather('../data/input/train_dayofweek_hour_rate.feather')


if __name__ == '__main__':
    main()