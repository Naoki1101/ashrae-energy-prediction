import pandas as pd
import numpy as np


def main():
    train = pd.read_feather('../data/input/train_and_leak.feather')

    train['date'] = train['timestamp'].apply(lambda x: x[:10])
    df = train.groupby(['building_id', 'meter', 'date'])['meter_reading'].sum().reset_index()

    for meter in df['meter'].unique():
        df[df['meter'] == meter].reset_index(drop=True).to_feather(f'../data/input/train_and_leak_meter{meter:.0f}.feather')


if __name__ == '__main__':
    main()