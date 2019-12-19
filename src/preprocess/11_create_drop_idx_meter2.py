import numpy as np
import pandas as pd
import joblib

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train_meter2.feather')

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

    joblib.dump(all_term_dict, '../pickle/zero_term_meter2.pkl')
    
    drop_idx = []
    for id_, term in all_term_dict.items():
        df = train[train['building_id'] == id_][train['date'] >= term[0]][train['date'] <= term[1]]
        drop_idx += list(df.index)

    extra_idx = [
        81983, 81984, 81985, 81991, 82002, 82003, 82004, 82005, 82006, 82007, 82008, 82036, 82054, 82074, 82075, 82085, 82086, 82127, 82131, 82133, 82135, 82137, 82139, 82145, 82147, 82148, 82149,
        82150, 82151,

        6476, 6477, 6478, 6479, 6480, 6481, 6482, 6483, 6484, 6485, 6486, 6487, 6488, 6489, 6490, 6491, 6492, 6493, 6494, 6495, 6496, 6497, 6498, 6499, 6500, 6501, 6502, 6503, 6504, 6505, 6506, 6507, 6508,
        6509, 6510, 6511, 6512, 6513, 6514, 6515, 6516, 6630, 6631, 6632, 6633, 6634,

        51946, 51947, 51948, 51949, 51974, 51998, 51999, 52007, 52008, 52009, 52010, 52011, 52012, 52013, 52014, 52015, 52016, 52017, 52018, 52019, 52020, 52021, 52022, 52023, 52024, 52025, 52026, 52027,
        52028, 52029, 52030, 52031, 52032, 52033, 52034, 52035, 52036, 52037, 52038, 52039, 52040, 52041, 52042, 52043, 52044, 52045, 52046, 52047, 52048, 52049, 52050, 52051, 52052, 52053, 52054, 52055,
        52056, 52057, 52058, 52059, 52060, 52061, 52062, 52063, 52064, 52065, 52066, 52067, 52068, 52069, 52070, 52071, 52072, 52073, 52074, 52075, 52076, 52077, 52078, 52079, 52080, 52081, 52082, 52083,
        52084, 52085, 52086, 52087, 52088, 52089, 52090, 52091, 52092, 52093, 52094, 52095, 52096, 52097, 52098, 52099, 52100, 52101, 52102, 52103, 52104, 52105, 52106, 52107, 52108, 52254, 52255,
    ]
        
    drop_idx += extra_idx
    
    np.save('../pickle/drop_idx_meter2.npy', np.array(drop_idx))

if __name__ == '__main__':
    main()