import argparse
import logging
import pandas as pd
import numpy as np
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
METER_TYPE = MODEL_NAME.split('_')[2]
RUN_NAME = f'{MODEL_NAME}_{NOW:%Y%m%d%H%M%S}'
SEED = config['compe']['seed']

DATA_PATH = Path('../data')
if 'leak' in MODEL_NAME:
    TRAIN_PATH = DATA_PATH / f'input/train_and_leak_{METER_TYPE}.feather'
else:
    TRAIN_PATH = DATA_PATH / f'input/train_{METER_TYPE}.feather'
SAMPLE_SUB_PATH = DATA_PATH / f'input/sample_submission_{METER_TYPE}.feather'

LOGGER_PATH = Path(f'../logs/{RUN_NAME}')

DATA_PARAMS = config['data']
FEATURES = DATA_PARAMS['features']
CATEGORICAL_FEATURES = DATA_PARAMS['categorical_features']
TARGET_NAME = config['compe']['target_name']
FOLD_NAME = config['data']['fold_name']

MODEL_PARAMS = config['model_params']
OOF_PARAMS = config['data']['oof']

NOTIFY_PARAMS = yml.load(options.notify)

REDUCE_FLAG = config['compe']['reduce']

# ===============
# Main
# ===============
t = Timer()
seed_everything(SEED)

LOGGER_PATH.mkdir()
logging.basicConfig(filename=LOGGER_PATH / 'train.log', level=logging.DEBUG)

yml.save(LOGGER_PATH /'config.yml', config)

with t.timer('load data and folds'):
    loader = DataLoader()
    train_x = loader.load_x(FEATURES, data_type='train', reduce=REDUCE_FLAG)
    test_x = loader.load_x(FEATURES, data_type='test', reduce=REDUCE_FLAG)
    train_y = loader.load_y(TRAIN_PATH, TARGET_NAME)
    folds = loader.load_folds(FOLD_NAME)

with t.timer('drop outliers'):
    if DATA_PARAMS['drop_fname'] is not None:
        drop_idx = np.load(f'../pickle/{DATA_PARAMS["drop_fname"]}')
        train_x = train_x.drop(drop_idx, axis=0).reset_index(drop=True)
        train_y = train_y.drop(drop_idx, axis=0).reset_index(drop=True)
        folds = folds.drop(drop_idx, axis=0).reset_index(drop=True)
    if OOF_PARAMS['concat_oof'] is not None:
        OOF_RUN_NAME = OOF_PARAMS['concat_oof']
        train_x['pred'] = np.load(f'../logs/{OOF_RUN_NAME}/oof.npy')
        test_x['pred'] = pd.read_csv(f'../logs/{OOF_RUN_NAME}/{METER_TYPE}.csv.gz')[TARGET_NAME]
        FEATURES += ['pred']

with t.timer('train and predict'):
    models, preds, oof, scores = train_and_predict(train_x, train_y, test_x,
                                                   MODEL_PARAMS,
                                                   folds,
                                                   model_name=MODEL_NAME,
                                                   cat_features=CATEGORICAL_FEATURES,
                                                   feval=None,
                                                   scoring_func=rmsle,
                                                   convert_type=config['data']['convert_type'])

    logging.disable(logging.FATAL)

    if OOF_PARAMS['save_oof']:
        np.save(f'../logs/{RUN_NAME}/oof.npy', oof)
        save_oof_plot(RUN_NAME, train_y, oof, type_='reg', dia=True)

with t.timer('save features importances'):
    save_importances(RUN_NAME, models, FEATURES)

with t.timer('make submission'):
    output_path = LOGGER_PATH / f'{METER_TYPE}.csv'
    make_submission(y_pred=np.mean(preds, axis=1), target_name=TARGET_NAME,
                    sample_path=SAMPLE_SUB_PATH, output_path=str(output_path), comp=True)

LOGGER_PATH.rename(f'../logs/{RUN_NAME}_{np.mean(scores):.3f}')

process_minutes = t.get_processing_time()

with t.timer('notify'):
    message = f'''{MODEL_NAME}\ncv: {np.mean(scores):.3f}\nscores: {scores}\ntime: {process_minutes:.2f}[min]'''

    send_line(NOTIFY_PARAMS['line']['token'], message)

    send_notion(token_v2=NOTIFY_PARAMS['notion']['token_v2'],
                url=NOTIFY_PARAMS['notion']['url'],
                name=RUN_NAME,
                created=NOW,
                model=MODEL_NAME.split('_')[0],
                local_cv=round(np.mean(scores), 4),
                time_=process_minutes,
                comment=COMMENT)
