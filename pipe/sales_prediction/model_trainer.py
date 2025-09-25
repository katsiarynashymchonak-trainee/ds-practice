import logging
import dill
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from config import PRED_PATH
from data_loader import DataLoader
from feature_selector import FeatureSelector
from hyperopt_tuner import HyperoptTuner
from pipe.sales_prediction.optuna_tuner import OptunaTuner
from standard_scaler_handler import StandardScalerHandler
from validator import TimeSeriesValidator
from feature_importance_evaluator import FeatureImportanceEvaluator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelTrainer:
    def __init__(self, models: tuple, metadata: dict = None, param_spaces: dict = None, use_feature_selection: bool = True):
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
        self.use_feature_selection = use_feature_selection
        self.selector = FeatureSelector() if use_feature_selection else None

    def unwrap_model(self, model): # if the type of model is hyperopt Pipeline
        if isinstance(model, Pipeline):
            return model.named_steps.get('regressor', model)
        return model

    def train(self, x, y):
        logger.info("Standardizing features before selection...")
        x = self.scaler.scale_train(x)

        if self.use_feature_selection:
            logger.info("Performing feature selection...")
            x = self.selector.fit_importance(x, y)

        for model in self.models:
            model_name = type(model).__name__
            logger.info(f"Evaluating model: {model_name}")

            if self.tuner_class and model_name in self.param_spaces:
                tuner = self.tuner_class(
                    model_class=type(model),
                    param_space=self.param_spaces[model_name],
                    scaler=self.scaler,
                    validator=self.validator,
                )
                tuner.tune(x, y)
                model = tuner.best_model
                tuned_error = tuner.best_score
                tuner.save_params_txt(f"data/best_params/{model_name}.txt")
            else:
                errors = self.validator.validate(model, x, y)
                tuned_error = errors.mean()

            # Unwrap pipeline to access feature importance
            final_model = self.unwrap_model(model)
            fimp_evaluator = FeatureImportanceEvaluator(model=final_model)
            fimp_evaluator.get_importance(x)

            logger.info(f"RMSE for {model_name}: {tuned_error:.4f}")

            if tuned_error < self.best_error:
                self.best_error = tuned_error
                self.best_model = model
                self.metadata.update({
                    "type": model_name,
                    "rmse": round(tuned_error, 4)
                })

        logger.info(f"Best model selected: {self.metadata['type']} with RMSE: {self.metadata['rmse']:.4f}")
        self.best_model.fit(x, y)

    def evaluate_best_model(self, x, y):
        logger.info("Evaluating best model on full dataset with cross-validation...")
        # Run cross-validation
        errors = self.validator.validate(self.best_model, x, y)
        mean_rmse = errors.mean()

        logger.info(f"Cross-validated RMSEs: {errors}")
        logger.info(f"Mean RMSE on full dataset: {mean_rmse:.4f}")

        return mean_rmse

    def predict(self, x_test):
        logger.info("Generating predictions...")

        if self.use_feature_selection:
            logger.info("Applying selected features to test set...")
            x_test = self.selector.transform(x_test)

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
        sub_df.to_csv(self.pred_path, index=False)

        logger.info(f"Predictions saved to {self.pred_path}")
        logger.info(f"First 5 predictions:\n{sub_df.head()}")

        return predictions

    def save(self, path: str = "event_pipe.pkl"):
        logger.info(f"Saving trained model to {path}...")
        with open(path, 'wb') as file:
            dill.dump({
                "model": self.best_model,
                "metadata": self.metadata
            }, file)
