import numpy as np
import pandas as pd
import joblib

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train_meter0.feather')

    # building_id = 1099
    drop_idx1 = train[train['building_id'] == 1099].index.values

    # building_id <= 104 & meter == 0 & 2016/05/20以前
    drop_idx2 = train[train['building_id'] <= 104]\
                     [train['meter'] == 0]\
                     [train['date'] <= '2016-05-20'].index.values
    
    drop_idx = np.sort(np.concatenate([drop_idx1, drop_idx2]))

    np.save('../pickle/drop_idx_meter0.npy', drop_idx)


if __name__ == '__main__':
    main()