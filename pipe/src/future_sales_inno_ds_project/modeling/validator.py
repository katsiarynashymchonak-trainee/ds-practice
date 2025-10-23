from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ..config import LOG_TARGET


class TimeSeriesValidator:
    def __init__(self, n_splits=4, regr_error=None):
        self.n_splits = n_splits
        self.regr_error = regr_error or self._rmse
        self.fold_metrics = []
        self.error_log = pd.DataFrame()

    @staticmethod
    def _rmse(y_true, y_pred):
        return sqrt(mean_squared_error(y_true, y_pred))

    def validate(self, model, x, y0):
        y = y0.copy()
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        error_records = []
        metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(x)):
            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Ensure y_train and y_val are Series
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.iloc[:, 0]
            if isinstance(y_val, pd.DataFrame):
                y_val = y_val.iloc[:, 0]

            # Reshape for LSTM if needed
            if hasattr(model, "fit") and "keras" in str(type(model)).lower():
                x_train_reshaped = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
                x_val_reshaped = x_val.values.reshape((x_val.shape[0], 1, x_val.shape[1]))

                model.fit(x_train_reshaped, y_train.values, epochs=10, batch_size=32, verbose=0)
                y_pred = model.predict(x_val_reshaped).flatten()
            else:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_val)

            # Ensure y_pred is 1D Series with correct index
            if isinstance(y_pred, np.ndarray):
                y_pred = np.squeeze(y_pred)
            y_pred = pd.Series(y_pred, index=x_val.index)

            # Inverse transform if target was log-transformed
            if LOG_TARGET:
                y_val = np.expm1(y_val)
                y_pred = np.expm1(y_pred)

            # Compute error metric
            fold_rmse = self.regr_error(y_val, y_pred)
            self.fold_metrics.append(fold_rmse)
            metrics.append(fold_rmse)

            # Create detailed error log
            fold_df = pd.DataFrame({
                'timestamp': x_val.index,
                'fold': fold,
                'y_true': y_val,
                'y_pred': y_pred,
                'abs_error': np.abs(y_val - y_pred),
                'error': y_val - y_pred,
                'target_magnitude': np.abs(y_val),
                'target_dynamic': np.abs(np.gradient(y_val.values))
            })

            error_records.append(fold_df)

        self.error_log = pd.concat(error_records).reset_index(drop=True)
        return np.array(metrics)

