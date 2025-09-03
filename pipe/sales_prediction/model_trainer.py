# model_trainer.py

import logging
import dill
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    def __init__(self, models: tuple, metadata: dict = None):
        self.models = models
        self.metadata = metadata or {}
        self.best_model = None
        self.best_score = float("-inf")

    def train(self, x, y, x_test):
        logger.info("Starting model training with time series cross-validation...")

        for model in self.models:
            model_name = type(model).__name__
            logger.info(f"Evaluating model: {model_name}")

            tscv = TimeSeriesSplit(n_splits=4)

            # Проверка на пропущенные значения
            if np.any(x.isnull()) or np.any(x_test.isnull()):
                pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('regressor', model)
                ])
            else:
                pipe = Pipeline([
                    ('regressor', model)
                ])

            # Кросс-валидация
            scores = cross_val_score(pipe, x, y, cv=tscv, scoring="neg_root_mean_squared_error")
            mean_score = scores.mean()
            std_score = scores.std()

            logger.info(f"RMSE for {model_name}: {-mean_score:.4f} ± {std_score:.4f}")

            # Выбор лучшей модели
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = pipe
                self.metadata.update({
                    "type": model_name,
                    "rmse": round(-mean_score, 4)
                })

        logger.info(f"Best model selected: {self.metadata['type']} with RMSE: {self.metadata['rmse']:.4f}")
        self.best_model.fit(x, y)

    def save(self, path: str = "event_pipe.pkl"):
        logger.info(f"Saving trained model to {path}...")
        with open(path, 'wb') as file:
            dill.dump({
                "model": self.best_model,
                "metadata": self.metadata
            }, file)
