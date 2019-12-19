import pandas as pd
import numpy as np


def main():
    train = pd.read_feather('../data/input/train.feather')

    train['date'] = train['timestamp'].apply(lambda x: x[:10])
    df = train.groupby(['building_id', 'meter', 'date'])['meter_reading'].sum().reset_index()

    for meter in df['meter'].unique():
        df[df['meter'] == meter].reset_index(drop=True).to_feather(f'../data/input/train_meter{meter}.feather')

    test = pd.read_feather('../data/input/test.feather')

    test['date'] = test['timestamp'].apply(lambda x: x[:10])
    df = test[~test[['building_id', 'meter', 'date']].duplicated()]
    for meter in df['meter'].unique():
        df_ = df[df['meter'] == meter]
        df_.reset_index(drop=True).to_feather(f'../data/input/test_meter{meter}.feather')
        
        sample_submission = pd.DataFrame({
            'building_id': df_['building_id'].values,
            'date': df_['date'].values,
            'meter_reading': np.zeros(len(df_))
            })
        sample_submission.to_feather(f'../data/input/sample_submission_meter{meter}.feather')


if __name__ == '__main__':
    main()