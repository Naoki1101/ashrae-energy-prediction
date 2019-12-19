import numpy as np
import pandas as pd

def target_encoding(tr, te, target, feat, fold):

    target_tr = np.zeros(len(tr))
    target_te = np.zeros(len(te))

    le = tr.groupby(feat)[target].mean().to_dict()
    target_te = te[feat].map(le).values

    for fold_ in fold['fold_id'].unique():
        X_tr = tr[fold['fold_id'] != fold_]
        X_val = tr[fold['fold_id'] == fold_]
        le = X_tr.groupby(feat)[target].mean().to_dict()
        target_tr[X_val.index] = X_val[feat].map(le).values

    return target_tr, target_te


def aggregation(df, id_name, numerical_name, aggs):
    df_agg = df.groupby(id_name)[numerical_name].agg(aggs).reset_index()
    agg_col_name = [f'{m}_{numerical_name}_each_{id_name}' for m in aggs]
    df_agg.columns = [id_name] + agg_col_name
    df = pd.merge(df, df_agg, on=id_name, how='left')
    return df[agg_col_name]
