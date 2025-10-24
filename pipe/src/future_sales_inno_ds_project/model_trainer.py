# Model trainer

import logging
import time
import dill
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

from .config import (
    PRED_PATH,
    USE_FEATURE_SELECTION,
    USE_OPTUNA,
    USE_PARALLEL_TUNING,
    MODEL_TYPE, BEST_PARAMS_PATH
)
from .data_preparation.data_loader import DataLoader
from .data_preparation.standard_scaler_handler import StandardScalerHandler
from .feature_engineering.feature_selector import FeatureSelector
from .model_interpetation.model_interpreter import ModelInterpreter
from .modeling.hyperopt_tuner import HyperoptTuner
from .modeling.optuna_tuner import OptunaTuner
from .modeling.validator import TimeSeriesValidator
from .modeling.model_factory import ModelFactory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    def __init__(self, model_type: str, spaces: dict = None, metadata: dict = None, neptune_run=None):
        # Initialize internal state and configuration
        self.metadata = metadata or {}
        self.best_model = None
        self.best_error = float("inf")
        self.validator = TimeSeriesValidator(n_splits=4)
        self.scaler = StandardScalerHandler()
        self.loader = DataLoader()
        self.pred_path = PRED_PATH
        self.param_spaces = spaces
        self.selector = FeatureSelector() if USE_FEATURE_SELECTION else None
        self.tuner_class = OptunaTuner if USE_OPTUNA else HyperoptTuner
        self.neptune_run = neptune_run
        if self.neptune_run:
            self.neptune_run["config/USE_FEATURE_SELECTION"] = USE_FEATURE_SELECTION
            self.neptune_run["config/USE_PARALLEL_TUNING"] = USE_PARALLEL_TUNING

        # Create model instance using factory pattern
        self.model_type = model_type
        self.model = ModelFactory.create(model_type)

    @staticmethod
    def unwrap_model(model):
        # Extract underlying model from pipeline if wrapped
        if isinstance(model, Pipeline):
            return model.named_steps.get('regressor', model)
        return model

    def train(self, x, y):
        # Main training entry point
        start = time.time()
        logger.info("Starting model training...")
        logger.info(f"Parallel tuning: {'enabled' if USE_PARALLEL_TUNING else 'disabled'}")

        # Preprocess training data
        x = self._prepare_training_data(x, y)
        logger.info(f"Training data shape after preprocessing: {x.shape}")
        logger.info(f"Evaluating model: {self.model_type}")

        # Special handling for LSTM models
        if self.model_type == "LSTM":
            tuned_model = self.model
            tuned_error = float("inf")
            self.best_model = tuned_model
            self.best_error = tuned_error
            self.metadata.update({
                "type": self.model_type,
                "rmse": tuned_error
            })
        else:
            # Tune or validate model
            tuned_model, tuned_error = self._tune_or_validate(self.model, self.model_type, x, y)
            final_model = self.unwrap_model(tuned_model)
            logger.info(f"RMSE for {self.model_type}: {tuned_error:.4f}")
            self._update_best_model(final_model, tuned_error, self.model_type)
            print(f"Best model selected: {self.metadata['type']} with RMSE: {self.metadata['rmse']:.4f}")

        logger.info(f"Actual class of best_model: {type(self.best_model)}")

        # Final model fitting
        if self.model_type == "LSTM":
            logger.info("Reshaping input for LSTM...")
            x_np = np.asarray(x, dtype=np.float32)
            y_np = np.asarray(y, dtype=np.float32)
            logger.info(f"x_np shape before reshape: {x_np.shape}")
            x_lstm = x_np.reshape((x_np.shape[0], 1, x_np.shape[1]))
            self.best_model.fit(x_lstm, y_np, epochs=10, batch_size=32, verbose=0)
        else:
            self.best_model.fit(x, y)

        logger.info(f"Model training completed in {time.time() - start:.2f}s")

    def _prepare_training_data(self, x, y):
        # Apply feature selection or scaling
        x_selected = x.copy()
        if USE_FEATURE_SELECTION:
            logger.info("Performing feature selection...")
            x_selected = self.selector.fit_importance(x_selected, y)
            return x_selected

        logger.info("Standardizing features...")
        x_scaled = self.scaler.scale_train(x_selected)
        return x_scaled

    def _tune_or_validate(self, model, model_name, x, y):
        # Tune model using Optuna/Hyperopt or validate directly
        if model_name == "LSTM":
            return model, float("inf")

        if self.tuner_class and model_name in self.param_spaces:
            tuner = self.tuner_class(
                model_class=type(model),
                param_space=self.param_spaces[model_name],
                scaler=self.scaler,
                validator=self.validator,
            )
            tuner.tune(x, y)
            tuner.save_params_txt(BEST_PARAMS_PATH / "{model_name}.txt")
            return tuner.best_model, tuner.best_score
        else:
            errors = self.validator.validate(model, x, y)
            return model, errors.mean()

    def _update_best_model(self, model, error, model_name):
        # Update best model if current one is better
        if error < self.best_error:
            self.best_error = error
            self.best_model = model
            self.metadata.update({
                "type": model_name,
                "rmse": round(error, 4)
            })

    def evaluate_best_model(self, x, y):
        # Evaluate best model with cross-validation
        start = time.time()
        logger.info("Evaluating best model with cross-validation...")
        errors = self.validator.validate(self.best_model, x, y)
        mean_rmse = errors.mean()

        logger.info(f"Cross-validated RMSEs: {errors}")
        logger.info(f"Mean RMSE: {mean_rmse:.4f}")
        logger.info(f"Evaluation completed in {time.time() - start:.2f}s")

        # Log RMSE per fold to Neptune
        if self.neptune_run:
            for i, fold_rmse in enumerate(errors):
                self.neptune_run[f"metrics/rmse/fold_{i + 1}"].append(fold_rmse)
            self.neptune_run["metrics/rmse/mean"] = mean_rmse

        # Run model interpretation
        interpreter = ModelInterpreter(
            model=self.best_model,
            model_type=MODEL_TYPE,
            validator=self.validator,
            neptune_run=self.neptune_run
        )
        interpreter.analyze()
        interpreter.explain(x)

        return mean_rmse

    def predict(self, x_test):
        # Generate predictions and save submission file
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

        if self.model_type == "LSTM":
            x_test_processed = x_test_processed.to_numpy()
            x_test_processed = x_test_processed.reshape(
                (x_test_processed.shape[0], 1, x_test_processed.shape[1])
            )

        predictions = self.best_model.predict(x_test_processed).flatten()

        sub_df = self.loader.load_submission_file()
        sub_df["item_cnt_month"] = predictions
        logger.info(f"Saving Predictions to {self.pred_path}")
        sub_df.to_csv(self.pred_path, index=False)

        logger.info(f"First 5 predictions: \n{sub_df.head()}")
        logger.info(f"Prediction completed in {time.time() - start:.2f}s")

        return predictions

    def save(self, path: str):
        # Save trained model and metadata to disk
        start = time.time()
        self.neptune_run["artifacts/predictions"].upload(path)
        logger.info(f"Saving trained model to {path}...")
        with open(path, 'wb') as file:
            dill.dump({
                "model": self.best_model,
                "metadata": self.metadata
            }, file)
        logger.info(f"Model saved successfully in {time.time() - start:.2f}s")
