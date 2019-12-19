import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train_meter0.feather')
    test = pd.read_feather('../data/input/test_meter0.feather')
    meta = pd.read_feather('../data/input/building_metadata.feather')

    df = pd.merge(train, meta, on='building_id', how='left')
    idx = df[df['site_id'] == 0].index.values
    np.save('../pickle/train_meter0_and_site0_idx.npy', idx)

    df = pd.merge(test, meta, on='building_id', how='left')
    idx = df[df['site_id'] == 0].index.values
    np.save('../pickle/test_meter0_and_site0_idx.npy', idx)



if __name__ == '__main__':
    main()