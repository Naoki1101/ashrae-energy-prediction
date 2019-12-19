import numpy as np
import pandas as pd
import joblib

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train_meter0.feather')

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

    joblib.dump(all_term_dict, '../pickle/zero_term_meter0.pkl')
    
    drop_idx = []
    for id_, term in all_term_dict.items():
        df = train[train['building_id'] == id_][train['date'] >= term[0]][train['date'] <= term[1]]
        drop_idx += list(df.index)

    extra_idx = [
        338789,
        464551, 464552, 464553, 464554, 464589, 464590, 464591, 464592, 464606, 464641, 464642, 464643, 464645, 464646, 464655, 464656, 464657, 464659, 464660, 464661, 464662, 464687, 464688, 464689,
        194953, 194959, 194960, 194961, 194965, 194966, 194972, 194973, 194974, 194975, 194976, 194977, 194978, 194979, 194980,
        465721, 465722, 465723, 465724, 465727, 465728, 465729, 465730, 465731, 465732, 465733, 465734,
        466398, 466399, 466400, 466401, 466402, 466403, 466404, 466405, 466406, 466407, 466408, 466409, 466410, 466411, 466531, 466532,
        101332, 101549, 101550, 101551, 101552, 101553, 101554, 101555, 101556,
    ]
        
    drop_idx += extra_idx
    
    np.save('../pickle/drop_idx_meter0.npy', np.array(drop_idx))

if __name__ == '__main__':
    main()