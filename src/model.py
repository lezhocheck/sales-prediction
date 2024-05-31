from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Any
import polars as pl
import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
import pickle
import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error,
    explained_variance_score,
    root_mean_squared_error
)


class Predictor(metaclass=ABCMeta):
    OPTIMIZATION_STEPS: int = 10
    OPTIMIZATION_RAND: int = 42
    OPTIMIZATION_LOSS: str = 'neg_mean_absolute_error'

    _estimator: Optional[BaseEstimator] = None

    def __init__(self, checkpoint: Optional[str] = None) -> None:
        if not checkpoint:
            return
        path = os.path.abspath(checkpoint)
        if not os.path.isfile(path):
            raise ValueError(f'Checkpoint {path} is not valid')
        self._estimator = pickle.load(open(path, 'rb'))

    def train(self, X: pl.DataFrame, y: pl.Series, force: bool = False) -> None:
        assert len(X) == len(y), 'Lengths of X and y are different'
        if self._estimator is None or force:
            if force:
                print('Model already trained. Starting retraining...')
            self._train(X, y)
        else:
            print('Model already trained. Skipping...')

    @abstractmethod
    def _train(self, X: pl.DataFrame, y: pl.Series) -> None:
        raise NotImplementedError()
    
    def predict(self, X: pl.DataFrame) -> pl.Series:
        if self._estimator is None:
            raise ValueError('Model was not fitted')
        return pl.Series(name='prediction', values=self._estimator.predict(X))
    
    def predict_single_sample(self, sample: Dict[str, Any]) -> float:
        if self._estimator is None:
            raise ValueError('Model was not fitted')
        X = pl.DataFrame(sample).select(self._estimator.feature_names_in_)
        return self._estimator.predict(X)[0]

    def eval(self, X: pl.DataFrame, y: pl.Series, apply_cv: bool = True) -> Dict[str, float]:
        if self._estimator is None:
            raise ValueError('Model was not fitted')
        if apply_cv:
            scores = cross_validate(self._estimator, X, y, cv=3, n_jobs=-1, scoring=[
                'neg_root_mean_squared_error', 
                'neg_mean_absolute_error',
                'explained_variance',
            ])
            return {k: np.mean(v) for k, v in scores.items() if not k.endswith('time')}
        predicted = self._estimator.predict(X)
        return {
            'test_neg_root_mean_squared_error': -root_mean_squared_error(y, predicted), 
            'test_neg_mean_absolute_error': -mean_absolute_error(y, predicted), 
            'test_explained_variance': explained_variance_score(y, predicted)
        }
    
    def save(self, path: str) -> None:
        if self._estimator is None:
            raise ValueError('Model was not fitted')
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):
            print(f'Overriding {path}')
        else:
            print(f'Saving to {path}')
        pickle.dump(self._estimator, open(path, 'wb'))


class RandomForestPredictor(Predictor):
    def _train(self, X: pl.DataFrame, y: pl.Series) -> None:
        def objective(trial: optuna.Trial) -> None:
            reg = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 2000),
                max_depth=trial.suggest_int('max_depth', 2, 32, log=True),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 16),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 16),
                bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                random_state=self.OPTIMIZATION_RAND
            )
            loss = cross_val_score(reg, X, y, n_jobs=-1, cv=3, scoring=self.OPTIMIZATION_LOSS)
            return loss.mean()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_jobs=-1, n_trials=self.OPTIMIZATION_STEPS)
        best_params = study.best_params
        self._estimator = RandomForestRegressor(**best_params, n_jobs=-1)
        self._estimator = self._estimator.fit(X, y)
    

class LgbmPredictor(Predictor):
    def _train(self, X: pl.DataFrame, y: pl.Series) -> None:
        def objective(trial: optuna.Trial) -> None:
            reg = lgb.LGBMRegressor(
                objective='regression',
                boosting_type='gbt',
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                learning_rate=trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                num_leaves=trial.suggest_int('num_leaves', 2, 256),
                max_depth=trial.suggest_int('max_depth', 1, 64),
                min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 10, 100),
                lambda_l1=trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                lambda_l2=trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True)
            )
            loss = cross_val_score(reg, X, y, n_jobs=-1, cv=3, scoring=self.OPTIMIZATION_LOSS)
            return loss.mean()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_jobs=-1, n_trials=self.OPTIMIZATION_STEPS)
        best_params = study.best_params
        self._estimator = lgb.LGBMRegressor(**best_params, n_jobs=-1)
        self._estimator = self._estimator.fit(X, y)