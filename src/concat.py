import argparse
import logging
import numpy as np
import pandas as pd
import datetime
import time
import yaml

from pathlib import Path

from utils import Timer, seed_everything
from utils import DataLoader, Yml, make_submission
from utils import send_line, send_notion
from runner import train_and_predict, save_importances, save_oof_plot
from metrics import rmsle

import warnings
warnings.filterwarnings('ignore')


# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument('--common', default='../configs/common/default.yml')
parser.add_argument('--notify', default='../configs/common/notify.yml')
parser.add_argument('-m', '--model')
parser.add_argument('-c', '--comment')
options = parser.parse_args()

yml = Yml()
config = yml.load(options.common)
config.update(yaml.load(open(f'../configs/exp/{options.model}.yml', 'r')))


# ===============
# Constants
# ===============
COMMENT = options.comment
NOW = datetime.datetime.now()
MODEL_NAME = options.model
RUN_NAME = f'{MODEL_NAME}_{NOW:%Y%m%d%H%M%S}'

DATA_PATH = Path('../data')
TRAIN_PATH = DATA_PATH / f'input/train.feather'
TEST_PATH = DATA_PATH / f'input/test.feather'
SAMPLE_SUB_PATH = DATA_PATH / f'input/sample_submission.feather'

TARGET_NAME = config['compe']['target_name']
FNAMES = config['fname']

LOGGER_PATH = Path(f'../logs/{RUN_NAME}')

# OOF_PARAMS = config['data']['oof']

NOTIFY_PARAMS = yml.load(options.notify)


# ===============
# Main
# ===============
t = Timer()

LOGGER_PATH.mkdir()

yml.save(LOGGER_PATH /'config.yml', config)

with t.timer('load data'):
    test = pd.read_csv('../data/input/test.csv')
    test['date'] = test['timestamp'].apply(lambda x: x[:10])
    test['hour'] = test['timestamp'].apply(lambda x: int(x[11:13]))

    if 'leak' in MODEL_NAME:
        df_rate = pd.read_feather(DATA_PATH / 'input/train_and_leak_hour_rate.feather')
    else:
        df_rate = pd.read_feather(DATA_PATH / 'input/train_hour_rate.feather')

with t.timer('calculate cv'):
    cv_list = []
    len_list = []
    for meter in range(4):
        meter_run_name = FNAMES[f'meter{meter}']
        cv_list.append(float(meter_run_name.split('_')[-1]))
        oof = np.load(f'../logs/{meter_run_name}/oof.npy')
        len_list.append(oof.shape[0])
    
    sle = 0
    for cv_, len_ in zip(cv_list, len_list):
        sle += (cv_ ** 2 * len_)
    
    cv = np.round(np.sqrt(sle / sum(len_list)), 5)

    print('\n\n===================================\n')
    print(f'CV: {cv}')
    print('\n===================================\n\n')

with t.timer('concat daily_predict'):
    df_predict_daily = pd.DataFrame(columns=['building_id', 'date', 'meter_reading', 'meter'])
    for meter in range(4):
        meter_run_name = FNAMES[f'meter{meter}']
        df = pd.read_csv(f'../logs/{meter_run_name}/meter{meter}.csv.gz')
        df['meter'] = meter
        
        df_predict_daily = pd.concat([df_predict_daily, df], axis=0)
    df_predict_daily.rename(columns={'meter_reading': 'meter_reading_daily'}, inplace=True)

with t.timer('calculate pred'):
    test = pd.merge(test, df_predict_daily, on=['building_id', 'date', 'meter'], how='left')
    test = pd.merge(test, df_rate, on=['building_id', 'meter', 'hour'], how='left')
    test['pred'] = test['meter_reading_daily'] * test['rate']
    pred = test['pred'].fillna(0)

with t.timer('replace with zero'):
    replace_id = [
        (1345, 0), 
        (778, 1), (1022, 1),
        (758, 2), (762, 2),
        (163, 3), (200, 3), (279, 3)
    ]
    for id_, m in replace_id:
        idx = test[test['building_id'] == id_][test['meter'] == m].index
        if len(idx) > 0:
            pred.iloc[idx] = 0

# with t.timer('replace with zero'):
#     replace_id = [
#         (803, 0), (857, 0), (1264, 0), (1345, 0), 
#         (778, 1), (780, 1), (1013, 1), (1018, 1), (1022, 1), (1093, 1), (1098, 1)
#         (758, 2), (762, 2), (1099, 2),
#         (163, 3), (200, 3), (279, 3), (287, 3)
#     ]
#     for id_, m in replace_id:
#         idx = test[test['building_id'] == id_][test['meter'] == m].index
#         if len(idx) > 0:
#             pred.iloc[idx] = 0

with t.timer('replace with leak'):
    leak = pd.read_feather(DATA_PATH / 'input/leak.feather')
    leak['timestamp'] = leak['timestamp'].astype(str)
    leak.rename(columns={'meter_reading': 'leak_meter_reading'}, inplace=True)

    test_and_leak = pd.merge(test, leak, on=['building_id', 'meter', 'timestamp'], how='left')
    leak_idx = test_and_leak['leak_meter_reading'].dropna().index
    pred.iloc[leak_idx] = test_and_leak.loc[leak_idx, 'leak_meter_reading']

with t.timer('make submission'):
    output_path = str(DATA_PATH / f'output/sub_{RUN_NAME}_{cv}.csv')
    make_submission(y_pred=pred, 
                    target_name=TARGET_NAME, 
                    sample_path=SAMPLE_SUB_PATH,
                    output_path=output_path,
                    comp=True)


# LOGGER_PATH.rename(f'../logs/{RUN_NAME}_{np.mean(scores):.3f}')

process_minutes = t.get_processing_time()

with t.timer('notify'):
    message = f'''{MODEL_NAME}\ncv: {cv:.3f}\nscores: \ntime: {process_minutes:.2f}[min]'''

    send_line(NOTIFY_PARAMS['line']['token'], message)

    send_notion(token_v2=NOTIFY_PARAMS['notion']['token_v2'],
                url=NOTIFY_PARAMS['notion']['url'],
                name=RUN_NAME,
                created=NOW,
                model='else',
                local_cv=cv,
                time_=process_minutes,
                comment=COMMENT)