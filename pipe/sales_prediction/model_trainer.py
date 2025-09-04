# Handles logging setup
import logging

import dill
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Custom time series validator
from validator import TimeSeriesValidator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    # Initializes with a list of models and optional metadata
    def __init__(self, models: tuple, metadata: dict = None):
        self.models = models
        self.metadata = metadata or {}
        self.best_model = None
        self.best_error = float("inf")  # Note: lower RMSE is better
        self.validator = TimeSeriesValidator(n_splits=4)

    # Trains and evaluates models using time series cross-validation
    def train(self, x, y, x_test):
        logger.info("Starting model training with custom time series validation...")

        for model in self.models:
            model_name = type(model).__name__
            logger.info(f"Evaluating model: {model_name}")

            # Build pipeline with imputation if missing values are present
            if np.any(x.isnull()) or np.any(x_test.isnull()):
                pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('regressor', model)
                ])
            else:
                pipe = Pipeline([
                    ('regressor', model)
                ])

            # Validate model and compute error metrics
            errors = self.validator.validate(pipe, x, y)
            mean_error = errors.mean()
            std_error = errors.std()

            logger.info(f"RMSE for {model_name}: {mean_error:.4f} ± {std_error:.4f}")

            # Update best model
            if mean_error < self.best_error:
                self.best_error = mean_error
                self.best_model = pipe
                self.metadata.update({
                    "type": model_name,
                    "rmse": round(mean_error, 4)
                })

        logger.info(f"Best model selected: {self.metadata['type']} with RMSE: {self.metadata['rmse']:.4f}")
        self.best_model.fit(x, y)

    # Saves the trained model and metadata
    def save(self, path: str = "event_pipe.pkl"):
        logger.info(f"Saving trained model to {path}...")
        with open(path, 'wb') as file:
            dill.dump({
                "model": self.best_model,
                "metadata": self.metadata
            }, file)
