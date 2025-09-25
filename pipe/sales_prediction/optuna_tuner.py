import logging
import os
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from pipe.sales_prediction.config import BASE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class OptunaTuner:
    def __init__(self, model_class, param_space, scaler, validator,
                 n_trials=15, sample_size=500_000):
        self.model_class = model_class
        self.param_space = param_space  # функция, принимающая trial и возвращающая dict
        self.scaler = scaler
        self.validator = validator
        self.n_trials = n_trials
        self.sample_size = sample_size

        self.best_model = None
        self.best_params = None
        self.best_score = float("inf")
        self.study = None

    def get_time_series_sample(self, x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        n_splits = self.validator.n_splits if hasattr(self.validator, "n_splits") else 4
        tscv = TimeSeriesSplit(n_splits=n_splits)

        x_sample_list = []
        y_sample_list = []

        for fold_idx, (_, val_idx) in enumerate(tscv.split(x)):
            val_x = x.iloc[val_idx]
            sample_count = min(self.sample_size, len(val_x))
            sampled_indices = np.random.choice(val_x.index, size=sample_count, replace=False)

            x_sample_list.append(x.loc[sampled_indices])
            y_sample_list.append(y.loc[sampled_indices])

            logger.info(f"Fold {fold_idx + 1}: sampled {sample_count} rows")

        x_sample = pd.concat(x_sample_list)
        y_sample = pd.concat(y_sample_list)

        logger.info(f"Final sampled shape: x={x_sample.shape}, y={y_sample.shape}")
        return x_sample, y_sample["item_cnt_month"]

    def tune(self, x: pd.DataFrame, y: pd.Series):
        print(f"Starting Optuna tuning for {self.model_class.__name__}")
        logger.info(f"Input x shape: {x.shape}, y shape: {y.shape}")

        x_sample, y_sample = self.get_time_series_sample(x, y)

        def objective(trial):
            params = self.param_space(trial)
            model = self.model_class(**params)
            pipe = Pipeline([
                ('scale', FunctionTransformer(self.scaler.scale_train)),
                ('regressor', model)
            ])

            errors = self.validator.validate(pipe, x_sample, y_sample)
            score = errors.mean()

            logger.debug(f"Params: {params}, CV RMSE: {score:.4f}")

            if score < self.best_score:
                self.best_score = score
                self.best_model = pipe
                self.best_params = params
                logger.info(f"New best score: {score:.4f}")

            return score

        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective, n_trials=self.n_trials)

        print(f"BEST RMSE {self.model_class.__name__}: {self.best_score:.4f}")
        print(f"BEST PARAMS {self.model_class.__name__}: {self.best_params}")

    def save_params_txt(self, relative_path: str):
        full_path = os.path.join(BASE_DIR, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        logger.info(f"Saving best hyperparameters to {full_path}")
        with open(full_path, 'w') as f:
            f.write(f"Model: {self.model_class.__name__}\n")
            f.write(f"Best RMSE: {self.best_score:.4f}\n")
            f.write("Best Parameters:\n")
            for key, value in self.best_params.items():
                f.write(f"  {key}: {value}\n")
