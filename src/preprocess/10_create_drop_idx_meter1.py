import numpy as np
import pandas as pd
import joblib

import warnings
warnings.filterwarnings('ignore')

def main():
    train = pd.read_feather('../data/input/train_meter1.feather')

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

    joblib.dump(all_term_dict, '../pickle/zero_term_meter1.pkl')
    
    drop_idx = []
    for id_, term in all_term_dict.items():
        df = train[train['building_id'] == id_][train['date'] >= term[0]][train['date'] <= term[1]]
        drop_idx += list(df.index)

    extra_idx = [
        140726, 140727, 140728, 140729, 140730, 140731, 140732, 140733, 140734, 140735, 140736, 140737, 140738, 140739, 140740, 140741, 140742, 140743, 140746, 140747, 140748, 140749, 140750, 140751,
        140756, 140757, 140758, 140759, 140767, 140773, 140774, 140792, 140793, 140925, 140926, 140927, 140928, 140929, 140932, 140933, 140934, 140935, 140966, 140967, 140968, 140969, 140970, 140971,
        140972, 140973, 140974, 140975, 140976, 140977, 140978, 140979, 140980, 140981, 140982, 140983, 140984, 140985, 140986, 141010, 141012, 141018, 141019, 141020, 141021, 141024, 141025, 141026,
        141027, 141030, 141031, 141032, 141033, 141034, 141035, 141036, 141037, 141038, 141039, 141042, 141043, 141044, 141045, 141047, 141048, 141049, 141051, 141052, 141053, 141054, 141057, 141058, 141059,
        
        141093, 141094, 141095, 141096, 141097, 141098, 141099, 141100, 141101, 141102, 141103, 141104, 141105, 141106, 141107, 141108, 141109, 141112, 141113, 141114, 141115, 141116, 141117, 141122,
        141123, 141124, 141125, 141138, 141139, 141140, 141154, 141158, 141165, 141166, 141211, 141212, 141213, 141214, 141215, 141216, 141217, 141218, 141219, 141220, 141221, 141222, 141223, 141224,
        141225, 141226, 141257, 141258, 141375, 141377, 141383, 141384, 141385, 141386, 141389, 141390, 141391, 141392, 141395, 141396, 141397, 141398, 141399, 141400, 141401, 141402, 141403, 141404,
        141406, 141407, 141408, 141409, 141410, 141412, 141413, 141414, 141416, 141417, 141418, 141419, 141422, 141423, 141424,

        147319, 147397, 147398, 147399, 147515, 147516, 147517, 147518, 147519, 147520, 147521, 147522, 147523, 147524, 147525, 147526, 147527, 147528, 147529, 147530, 147531, 147532, 147533, 147534, 147535,

        124643, 124644, 124645, 124646, 124647, 124648, 124649, 124650, 124651, 124652, 124653, 124654, 124655, 124656, 124657, 124658, 124659, 124660, 124661, 124662, 124663,

        37873, 37874, 37875, 37876, 37877, 37878, 37879, 37880, 37881, 38050, 38051, 38061, 38062, 38063, 38064, 38069, 38070, 38071, 38072, 38073, 38079, 38080, 38081, 38082, 38086, 38087, 38088,

        28003, 28004, 28005, 28006, 28007, 28008, 28009, 28010, 28011, 28012, 28013, 28104, 28105,
    ]
        
    drop_idx += extra_idx
    
    np.save('../pickle/drop_idx_meter1.npy', np.array(drop_idx))

if __name__ == '__main__':
    main()