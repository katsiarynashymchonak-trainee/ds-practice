# validator.py

from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesValidator:
    def __init__(self, n_splits=4, regr_error=None):
        self.n_splits = n_splits
        self.regr_error = regr_error or self._rmse

    @staticmethod
    def _rmse(y_true, y_pred):
        return -sqrt(mean_squared_error(y_true, y_pred))

    def validate(self, model, x, y):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(x)):
            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            metric = -self.regr_error(y_val, y_pred)
            metrics.append(metric)

        return np.array(metrics)
