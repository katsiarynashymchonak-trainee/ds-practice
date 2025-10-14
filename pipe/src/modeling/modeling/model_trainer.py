# model_trainer.py

import logging
import time

import dill
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from pipe.src.config import PRED_PATH, USE_FEATURE_SELECTION
from pipe.src.data_preparation.data_loader import DataLoader
from pipe.src.data_preparation.standard_scaler_handler import StandardScalerHandler
from pipe.src.feature_engineering.feature_engineering.feature_importance_evaluator import FeatureImportanceEvaluator
from pipe.src.feature_engineering.feature_engineering.feature_selector import FeatureSelector
from pipe.src.modeling.modeling.optuna_tuner import OptunaTuner
from pipe.src.modeling.validator import TimeSeriesValidator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    def __init__(self, models: tuple, metadata: dict = None, param_spaces: dict = None,
                 use_feature_selection: bool = True):
        self.models = models
        self.metadata = metadata or {}
        self.best_model = None
        self.best_error = float("inf")
        self.validator = TimeSeriesValidator(n_splits=4)
        self.scaler = StandardScalerHandler()
        self.loader = DataLoader()
        self.pred_path = PRED_PATH
        self.tuner_class = OptunaTuner
        self.param_spaces = param_spaces or {}
        self.selector = FeatureSelector() if use_feature_selection else None

    # Unwrapping pipeline to access underlying model
    @staticmethod
    def unwrap_model(model):
        if isinstance(model, Pipeline):
            return model.named_steps.get('regressor', model)
        return model

    # Training all models and selecting the best one
    def train(self, x, y):
        start = time.time()
        logger.info("Starting model training...")

        x = self._prepare_training_data(x, y)
        logger.info(f"Training data shape after preprocessing: {x.shape}")

        for model in self.models:
            model_name = type(model).__name__
            logger.info(f"Evaluating model: {model_name}")

            tuned_model, tuned_error = self._tune_or_validate(model, model_name, x, y)

            final_model = self.unwrap_model(tuned_model)
            FeatureImportanceEvaluator(model=final_model).get_importance(x)

            logger.info(f"RMSE for {model_name}: {tuned_error:.4f}")
            self._update_best_model(final_model, tuned_error, model_name)

        print(f"Best model selected: {self.metadata['type']} with RMSE: {self.metadata['rmse']:.4f}")
        self.best_model.fit(x, y)
        logger.info(f"Model training completed in {time.time() - start:.2f}s")

    # Preparing training data: scaling and feature selection
    def _prepare_training_data(self, x, y):
        x_selected = x.copy()
        if USE_FEATURE_SELECTION:
            logger.info("Performing feature selection...")
            x_selected = self.selector.fit_importance(x_selected, y)
            return x_selected

        logger.info("Standardizing features...")
        x_scaled = self.scaler.scale_train(x_selected)
        return x_scaled

    # Tuning model with Optuna or validating directly
    def _tune_or_validate(self, model, model_name, x, y):
        if self.tuner_class and model_name in self.param_spaces:
            tuner = self.tuner_class(
                model_class=type(model),
                param_space=self.param_spaces[model_name],
                scaler=self.scaler,
                validator=self.validator,
            )
            tuner.tune(x, y)
            tuner.save_params_txt(f"data/best_params/{model_name}.txt")
            return tuner.best_model, tuner.best_score
        else:
            errors = self.validator.validate(model, x, y)
            return model, errors.mean()

    # Updating best model if current one is better
    def _update_best_model(self, model, error, model_name):
        if error < self.best_error:
            self.best_error = error
            self.best_model = model
            self.metadata.update({
                "type": model_name,
                "rmse": round(error, 4)
            })

    # Evaluating best model using cross-validation
    def evaluate_best_model(self, x, y):
        start = time.time()
        logger.info("Evaluating best model with cross-validation...")
        errors = self.validator.validate(self.best_model, x, y)
        mean_rmse = errors.mean()

        logger.info(f"Cross-validated RMSEs: {errors}")
        logger.info(f"Mean RMSE: {mean_rmse:.4f}")
        logger.info(f"Evaluation completed in {time.time() - start:.2f}s")

        return mean_rmse

    # Generating predictions and saving submission file
    def predict(self, x_test):
        start = time.time()
        logger.info("Generating predictions...")

        if USE_FEATURE_SELECTION:
            logger.info("Applying selected features to test set...")
            x_test = self.selector.transform(x_test)

        logger.info(f"Test data shape before scaling: {x_test.shape}")
        if np.any(x_test.isnull()):
            logger.info("Filling missing values in test set...")
            preprocess_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scale', FunctionTransformer(self.scaler.scale_test))
            ])
            x_test_processed = preprocess_pipe.fit_transform(x_test)
        else:
            x_test_processed = self.scaler.scale_test(x_test)

        predictions = self.best_model.predict(x_test_processed)

        sub_df = self.loader.load_submission_file()
        sub_df["item_cnt_month"] = predictions
        logger.info(f"Saving Predictions to {self.pred_path}")
        sub_df.to_csv(self.pred_path, index=False)

        logger.info(f"First 5 predictions: \n{sub_df.head()}")
        logger.info(f"Prediction completed in {time.time() - start:.2f}s")

        return predictions

    # Saving trained model and metadata to disk
    def save(self, path: str):
        start = time.time()
        logger.info(f"Saving trained model to {path}...")
        with open(path, 'wb') as file:
            dill.dump({
                "model": self.best_model,
                "metadata": self.metadata
            }, file)
        logger.info(f"Model saved successfully in {time.time() - start:.2f}s")
