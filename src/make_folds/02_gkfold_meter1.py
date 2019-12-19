import os
import sys
import pandas as pd
import numpy as np
import yaml

from validate_utils import FoldValidation

import warnings
warnings.filterwarnings('ignore')


with open('../configs/common/default.yml', 'r') as yf:
    config = yaml.load(yf)

# ===============
# Settings
# ===============
fname = os.path.basename(sys.argv[0])
INPUT_PATH = f'../data/input/train_meter1.feather'
OUTPUT_PATH = f'../folds/{fname.split(".")[0]}.feather'


# ===============
# Main
# ===============
df = pd.read_feather(INPUT_PATH)
fold_id = df['date'].apply(lambda x: (int(x[5:7]) - 1) // 3)
folds = pd.DataFrame({'fold_id': fold_id})
folds.to_feather(OUTPUT_PATH)