import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train.feather')
    train['date'] = train['timestamp'].apply(lambda x: x[:10])
    train['hour'] = train['timestamp'].apply(lambda x: int(x[11:13]))

    drop_idx = train[train['meter_reading'] == 0].index
    train = train.drop(drop_idx, axis=0).reset_index(drop=True)

    df_sum = train.groupby(['building_id', 'meter', 'date'])['meter_reading'].sum().reset_index()
    df_sum.rename(columns={'meter_reading': 'meter_reading_day'}, inplace=True)

    train_ = pd.merge(train, df_sum, on=['building_id', 'meter', 'date'], how='left')

    s = 0.001
    train_['rate'] = train_['meter_reading'] / (train_['meter_reading_day'] + s)

    df_rate = train_.groupby(['building_id', 'meter', 'hour'])['rate'].mean().reset_index()

    unique_building_id = np.sort(df_rate['building_id'].unique())
    unique_meter = np.sort(df_rate['meter'].unique())

    for id_ in unique_building_id:
        for m in unique_meter:
            df = df_rate[df_rate['building_id'] == id_][df_rate['meter'] == m]
            if len(df) > 0:
                rate_arr = df['rate'].values
                rate_arr_scaled = np.exp(rate_arr) / np.sum(np.exp(rate_arr))
                df_rate.loc[df.index, 'rate'] = rate_arr_scaled
    

    df_rate.reset_index(drop=True).to_feather('../data/input/train_hour_rate_v2.feather')


if __name__ == '__main__':
    main()