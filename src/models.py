from abc import ABCMeta, abstractmethod

from catboost import Pool, CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb

from lightgbm.callback import _format_eval_result
import logging


class Model(metaclass=ABCMeta):

    def __init__(self, params):
        self.params = params
        self.model = None

    @abstractmethod
    def fit(self, tr_x, tr_y, te_x, va_x=None, va_y=None, cat_features=None):
        pass

    @abstractmethod
    def predict(self, te_x, cat_features=None):
        pass


# ===============
# LightGBM
# ===============
class LGBClassifier(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=cat_features)
        if validation:
            lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=cat_features)

        logger = logging.getLogger('main')
        callbacks = [self.log_evaluation(logger, period=100)]

        if validation:
            self.model = lgb.train(self.params,
                                   lgb_train,
                                   num_boost_round=10000,
                                   valid_sets=[lgb_eval],
                                   verbose_eval=200,
                                   early_stopping_rounds=200,
                                   callbacks=callbacks,
                                   feval=feval)
        else:
            self.model = lgb.train(self.params,
                                   lgb_train,
                                   num_boost_round=10000,
                                   callbacks=callbacks)

        logging.debug(self.model.best_iteration)
        logging.debug(self.model.best_score['valid_1'][self.params['metric']])

    def predict(self, te_x, cat_features=None):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def extract_importances(self, imp_type='gain'):
        return self.model.feature_importance(importance_type=imp_type)  # imp_type = 'gain' or 'split'

    def log_evaluation(self, logger, period=1, show_stdv=True, level=logging.DEBUG):
        def _callback(env):
            if period > 0 and env.evaluation_result_list \
                    and (env.iteration + 1) % period == 0:
                result = '\t'.join([
                    _format_eval_result(x, show_stdv)
                    for x in env.evaluation_result_list
                ])
                logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
        _callback.order = 10
        return _callback


class LGBRegressor(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=cat_features)
        if validation:
            lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=cat_features)

        logger = logging.getLogger('main')
        callbacks = [self.log_evaluation(logger, period=100)]

        if validation:
            self.model = lgb.train(self.params,
                                   lgb_train,
                                   num_boost_round=10000,
                                   valid_sets=[lgb_eval],
                                   verbose_eval=200,
                                   early_stopping_rounds=200,
                                   callbacks=callbacks,
                                   feval=feval)
        else:
            self.model = lgb.train(self.params,
                                   lgb_train,
                                   num_boost_round=10000,
                                   callbacks=callbacks)

        logging.debug(self.model.best_iteration)
        logging.debug(self.model.best_score['valid_0'][self.params['metric']])

    def predict(self, te_x, cat_features=None):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def extract_importances(self, imp_type='gain'):  # imp_type = 'gain' or 'split'
        return self.model.feature_importance(importance_type=imp_type)

    def log_evaluation(self, logger, period=100, show_stdv=True, level=logging.DEBUG):
        def _callback(env):
            if period > 0 and env.evaluation_result_list \
                    and (env.iteration + 1) % period == 0:
                result = '\t'.join([
                    _format_eval_result(x, show_stdv)
                    for x in env.evaluation_result_list
                ])
                logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
        _callback.order = 10
        return _callback


# ===============
# Catboost
# ===============
class CBClassifier(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        for col in cat_features:
            tr_x[col] = tr_x[col].astype('category')
            va_x[col] = va_x[col].astype('category')

        validation = va_x is not None
        cb_train = Pool(tr_x, label=tr_y, cat_features=cat_features)
        if validation:
            cb_valid = Pool(va_x, label=va_y, cat_features=cat_features)

        cb = CatBoostClassifier(**self.params)

        if validation:
            self.model = cb.fit(cb_train,
                                eval_set=cb_valid,
                                use_best_model=True,
                                verbose_eval=200,
                                plot=False)
        else:
            self.model = cb.fit(cb_train)

        logging.debug(self.model.best_iteration_)
        logging.debug(self.model.best_score_['validation'][self.params['eval_metric']])

    def predict(self, te_x, cat_features=None):
        for col in cat_features:
            te_x[col] = te_x[col].astype('category')

        return self.model.predict(te_x)

    def extract_importances(self):
        return self.model.feature_importances_


class CBRegressor(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        for col in cat_features:
            tr_x[col] = tr_x[col].astype('category')
            va_x[col] = va_x[col].astype('category')

        validation = va_x is not None
        cb_train = Pool(tr_x, label=tr_y, cat_features=cat_features)
        if validation:
            cb_valid = Pool(va_x, label=va_y, cat_features=cat_features)

        cb = CatBoostRegressor(**self.params)

        if validation:
            self.model = cb.fit(cb_train,
                                eval_set=cb_valid,
                                use_best_model=True,
                                verbose_eval=200,
                                plot=False)
        else:
            self.model = cb.fit(cb_train)

        logging.debug(self.model.best_iteration_)
        logging.debug(self.model.best_score_['validation'][self.params['eval_metric']])

    def predict(self, te_x, cat_features=None):
        for col in cat_features:
            te_x[col] = te_x[col].astype('category')

        return self.model.predict(te_x)

    def extract_importances(self):
        return self.model.feature_importances_
