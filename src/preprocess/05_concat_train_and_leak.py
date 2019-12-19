import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train.feather')
    leak = pd.read_feather('../data/input/leak.feather')
    leak['timestamp'] = leak['timestamp'].astype(str)

    leak_test = leak[leak['timestamp'] >= '2017-01-01 00:00:00']

    train_and_leak = pd.concat([train, leak_test[train.columns]], axis=0).reset_index(drop=True)

    train_and_leak.to_feather('../data/input/train_and_leak.feather')


if __name__ == '__main__':
    main()