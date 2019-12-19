import numpy as np
import pandas as pd
import joblib

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train_meter3.feather')

    unique_building = np.sort(train['building_id'].unique())
    all_term_dict = {}

    for id_ in unique_building:
        df = train[train['building_id'] == id_]
        max_ = df['meter_reading'].max()
        zero_counter = 0
        max_zero_count = 0
        term = ['', '']
        idx_list = list(df.index)

        all_zero_count = len(df[df['meter_reading'] == 0])
        if all_zero_count > 0:
            for i, idx in enumerate(idx_list):
                target = df.loc[idx, 'meter_reading']
                date_ = df.loc[idx, 'date']
                if target < max_ / 20 and date_ != '2016-12-31':
                    if zero_counter == 0:
                        term_begin = df.loc[idx, 'date']
                    zero_counter += 1
                elif target != 0 and zero_counter > 0:
                    if zero_counter >= max_zero_count and zero_counter >= 10:
                        term[0] = term_begin
                        term[1] = df.loc[idx_list[i - 1], 'date']
                        max_zero_count = zero_counter
                    zero_counter = 0
                elif target == 0 and date_ == '2016-12-31' and zero_counter > 0:
                    if zero_counter >= max_zero_count and zero_counter >= 10:
                        term[0] = term_begin
                        term[1] = df.loc[idx_list[i], 'date']
                        max_zero_count = zero_counter
                    zero_counter = 0
                else:
                    pass
            
        all_term_dict[id_] = term

    joblib.dump(all_term_dict, '../pickle/zero_term_meter3.pkl')
    
    drop_idx = []
    for id_, term in all_term_dict.items():
        df = train[train['building_id'] == id_][train['date'] >= term[0]][train['date'] <= term[1]]
        drop_idx += list(df.index)

    extra_idx = [
        29047, 29048, 29049, 29050, 29051, 29052, 29053, 29054, 29055, 29056, 29060, 29061, 29062, 29063, 29064, 29065, 29066, 29067, 29068, 29069, 29071, 29139, 29140, 29150, 29151, 29167,

        40060, 40062, 40067, 40072, 40073, 40076, 40082, 40085, 40086, 40089, 40092, 40114, 40140, 40141, 40142,

        49939, 49940, 49941, 49942, 49945, 49946, 49947, 49948, 49949, 49950, 49951, 49952,
    ]
        
    drop_idx += extra_idx
    
    np.save('../pickle/drop_idx_meter3.npy', np.array(drop_idx))

if __name__ == '__main__':
    main()