import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train_and_leak.feather')
    train['date'] = train['timestamp'].apply(lambda x: x[:10])
    train['hour'] = train['timestamp'].apply(lambda x: int(x[11:13]))

    drop_idx = train[train['building_id'] <= 104][train['meter'] == 0][train['date'] <= '2016-05-20'].index
    train = train.drop(drop_idx, axis=0).reset_index(drop=True)

    df_sum = train.groupby(['building_id', 'meter', 'date'])['meter_reading'].sum().reset_index()
    df_sum.rename(columns={'meter_reading': 'meter_reading_day'}, inplace=True)

    train_ = pd.merge(train, df_sum, on=['building_id', 'meter', 'date'], how='left')

    s = 0.001
    train_['rate'] = train_['meter_reading'] / (train_['meter_reading_day'] + s)

    df_rate = train_.groupby(['building_id', 'meter', 'hour'])['rate'].mean().reset_index()

    df_rate.reset_index(drop=True).to_feather('../data/input/train_and_leak_hour_rate.feather')


if __name__ == '__main__':
    main()