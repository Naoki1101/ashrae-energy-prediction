import numpy as np
import pandas as pd
import logging

from models import LGBRegressor, LGBClassifier, CBRegressor, CBClassifier

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


def train_and_predict(train_x, train_y, test_x, params, folds, model_name=None,
                      cat_features=None, feval=None, scoring_func=None, convert_type='raw'):

    unique_fold = np.sort(folds['fold_id'].unique())
    if 'lgbm_reg' in model_name:
        model = LGBRegressor(params)
    elif 'lgbm_clf' in model_name:
        model = LGBClassifier(params)
    elif 'cb_reg' in model_name:
        model = CBRegressor(params)
    elif 'cb_clf' in model_name:
        model = CBClassifier(params)
    else:
        raise(NotImplementedError)

    print('MODEL: ', model.__class__.__name__)

    if convert_type == 'log':
        train_y = np.log1p(train_y)

    preds = np.zeros((len(test_x), len(unique_fold)))
    oof = np.zeros(len(train_x))
    scores = []
    models = []

    for fold_ in unique_fold:
        print(f'\n\nfold{fold_}')
        logging.debug(f'\n\nfold{fold_}')

        tr_x, va_x = train_x[folds['fold_id'] != fold_], train_x[folds['fold_id'] == fold_]
        tr_y, va_y = train_y[folds['fold_id'] != fold_], train_y[folds['fold_id'] == fold_]

        model.fit(tr_x, tr_y, va_x, va_y, cat_features=cat_features, feval=feval)

        va_pred = model.predict(va_x, cat_features)
        oof[va_x.index] = va_pred

        if convert_type == 'log':
            va_y = np.where(np.expm1(va_y) >= 0, np.expm1(va_y), 0)
            va_pred = np.where(np.expm1(va_pred) >= 0, np.expm1(va_pred), 0)
            score = scoring_func(va_y, va_pred)
        else:
            score = scoring_func(va_y, va_pred)

        scores.append(np.round(score, 3))
        print(f'\nScore: {score}')
        logging.debug(f'Score: {score}')

        pred = model.predict(test_x, cat_features)
        if convert_type == 'log':
            pred = np.where(np.expm１(pred) >= 0, np.expm1(pred), 0)

        preds[:, fold_] = pred

        models.append(model)

    if convert_type == 'log':
        oof = np.where(np.expm1(oof) >= 0, np.expm1(oof), 0)

    print('\n\n===================================\n')
    print(f'CV: {np.mean(scores)}')
    logging.debug(f'\n\nCV: {np.mean(scores)}\n\n')
    print('\n===================================\n\n')

    return models, preds, oof, scores


def save_importances(run_name, models, features):
    df_feature_importance = pd.DataFrame()
    for fold_, model in enumerate(models):

        if 'lgbm' in run_name:
            fold_importance = model.extract_importances(imp_type='gain')
        elif 'cb' in run_name:
            fold_importance = model.extract_importances()
        else:
            raise(NotImplementedError)

        df_fold_importance = pd.DataFrame()
        df_fold_importance['feature'] = features
        df_fold_importance['importance'] = fold_importance

        df_feature_importance = pd.concat([df_feature_importance, df_fold_importance], axis=0)

    df_unique_feature_importance = (df_feature_importance[['feature', 'importance']]
                                    .groupby('feature')
                                    .mean()
                                    .sort_values(by='importance', ascending=False))
    df_unique_feature_importance.to_csv(f'../logs/{run_name}/importances.csv', index=True)

    cols = df_unique_feature_importance.index
    df_best_features = df_feature_importance.loc[df_feature_importance['feature'].isin(cols)]

    plt.figure(figsize=(14, int(np.log(len(cols)) * 50)))
    sns.barplot(x='importance',
                y='feature',
                data=df_best_features.sort_values(by="importance",
                                                  ascending=False))

    plt.title(f'{run_name} Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(f'../logs/{run_name}/importances_plot.png')


def save_oof_plot(run_name, y_true, oof, type_='reg', dia=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    if type_ == 'reg':
        ax1.hist(oof, bins=50)
        ax2.scatter(oof, y_true)
        max_ = max(max(oof), max(y_true))
        min_ = min(min(oof), min(y_true))
        if dia:
            ax2.plot([min_, max_], [min_, max_], 'orange', '-')
        pad_ = min_ * 0.1
        ax2.set_xlim([min_ - pad_, max_ + pad_])
        ax2.set_ylim([min_ - pad_, max_ + pad_])

    plt.tight_layout()
    plt.savefig(f'../logs/{run_name}/oof_plot.png')
