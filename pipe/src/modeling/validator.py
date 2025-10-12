from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesValidator:
    def __init__(self, n_splits=4, regr_error=None):
        # Number of splits for time series cross-validation
        self.n_splits = n_splits

        # Error metric to evaluate predictions (default: RMSE)
        self.regr_error = regr_error or self._rmse

        # Stores RMSE for each fold
        self.fold_metrics = []

        # Stores detailed error records across all folds
        self.error_log = pd.DataFrame()

    @staticmethod
    def _rmse(y_true, y_pred):
        # Root Mean Squared Error as default metric
        return sqrt(mean_squared_error(y_true, y_pred))

    def validate(self, model, x, y):
        # Initialize time series cross-validator
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        error_records = []  # Collects per-fold error details
        metrics = []        # Stores signed metrics for return

        for fold, (train_idx, val_idx) in enumerate(tscv.split(x)):
            # Split data into training and validation sets
            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit model and generate predictions
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)

            # Compute error metric for current fold
            fold_rmse = self.regr_error(y_val, y_pred)
            self.fold_metrics.append(fold_rmse)

            metrics.append(-fold_rmse)

            # Create detailed error log for current fold
            fold_df = pd.DataFrame({
                'timestamp': x_val.index,                     # Time index
                'fold': fold,                                 # Fold number
                'y_true': y_val,                              # Actual values
                'y_pred': y_pred,                             # Predicted values
                'abs_error': np.abs(y_val - y_pred),          # Absolute error
                'error': y_val - y_pred,                      # Error
                'target_magnitude': np.abs(y_val),            # Magnitude of target
                'target_dynamic': np.abs(np.gradient(y_val))  # Change in target
            })
            error_records.append(fold_df)

        # Combine all fold logs into a single DataFrame
        self.error_log = pd.concat(error_records).reset_index(drop=True)

        # Return signed metrics as in original version
        return np.array(metrics)
