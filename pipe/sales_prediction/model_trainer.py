import logging

import dill
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sales_prediction.standard_scaler_handler import StandardScalerHandler
from sales_prediction.validator import TimeSeriesValidator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    def __init__(self, models: tuple, metadata: dict = None):
        self.models = models
        self.metadata = metadata or {}
        self.best_model = None
        self.best_error = float("inf")
        self.validator = TimeSeriesValidator(n_splits=4)
        self.scaler = StandardScalerHandler()

    def train(self, x, y):
        logger.info("Starting model training with custom time series validation...")

        for model in self.models:
            model_name = type(model).__name__
            logger.info(f"Evaluating model: {model_name}")

            if np.any(x.isnull()):
                pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ("scale", FunctionTransformer(self.scaler.scale_train)),
                    ('regressor', model)
                ])
            else:
                pipe = Pipeline([
                    ("scale", FunctionTransformer(self.scaler.scale_train)),
                    ('regressor', model)
                ])

            errors = self.validator.validate(pipe, x, y)
            mean_error = errors.mean()
            std_error = errors.std()

            logger.info(f"RMSE for {model_name}: {mean_error:.4f} ± {std_error:.4f}")

            if mean_error < self.best_error:
                self.best_error = mean_error
                self.best_model = pipe
                self.metadata.update({
                    "type": model_name,
                    "rmse": round(mean_error, 4)
                })

        logger.info(f"Best model selected: {self.metadata['type']} with RMSE: {self.metadata['rmse']:.4f}")
        self.best_model.fit(x, y)

    def predict(self, x_test):
        logger.info("Prediction...")

        if np.any(x_test.isnull()):
            logger.info("Filling NA in test...")
            preprocess_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scale', FunctionTransformer(self.scaler.scale_test))
            ])
            x_test_processed = preprocess_pipe.fit_transform(x_test)
        else:
            x_test_processed = self.scaler.scale_test(x_test)

        # Prediction
        predictions = self.best_model.predict(x_test_processed)

        # Output results
        logger.info(f"First 5 results: {predictions[:5]}")
        return predictions

    def save(self, path: str = "event_pipe.pkl"):
        logger.info(f"Saving trained model to {path}...")
        with open(path, 'wb') as file:
            dill.dump({
                "model": self.best_model,
                "metadata": self.metadata
            }, file)
