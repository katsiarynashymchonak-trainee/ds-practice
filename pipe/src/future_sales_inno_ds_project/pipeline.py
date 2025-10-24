import logging
import lightgbm
import neptune
import pandas as pd
import sklearn
import xgboost

from .config import (
    MODEL_SAVE_PATH, MODEL_METADATA, FORM_PREP_DATA,
    X_PATH, X_TEST_PATH, Y_PATH, OPTUNA_SPACES, HYPEROPT_SPACES, USE_OPTUNA, LOG_TARGET, MODEL_TYPE,
)
from .data_preparation.data_loader import DataLoader
from .data_preparation.data_preprocessor import DataPreprocessor
from .feature_engineering.feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from dotenv import load_dotenv
import getpass


def get_neptune_token():
    load_dotenv()
    token = os.getenv("NEPTUNE_API_TOKEN")
    if not token:
        token = getpass.getpass("Input NEPTUNE API token: ")
    return token


def initialize_neptune_run(api_token: str):
    return neptune.init_run(
        project="katsiaryna.shymchonak/Future-sales",
        api_token=api_token,
    )


def prepare_data(run):
    if FORM_PREP_DATA:
        # Load raw input data
        loader = DataLoader()
        train, items, categories, shops, test = loader.load_data()

        transformer = sklearn.pipeline.FunctionTransformer
        # Build preprocessing pipeline
        pipeline = sklearn.pipeline.Pipeline([
            ("aggregate_data", transformer(DataPreprocessor.aggregate_data)),
            ("create_full_train", transformer(FeatureEngineer.create_full_train)),
            ("remove_shops", transformer(FeatureEngineer.remove_shops_from_train, kw_args={"test": test})),
            ("remove_zero_sales", transformer(DataPreprocessor.remove_zero_sales)),
            ("concat_data", transformer(FeatureEngineer.concat_train_test, kw_args={"test": test})),
            ("replace_shop_ids", transformer(DataPreprocessor.replace_shop_ids)),
            ("add_features", transformer(FeatureEngineer.add_features,
                                         kw_args={"items": items,
                                                  "categories": categories,
                                                  "shops": shops})),
            ("cast_types", transformer(DataPreprocessor.cast_types))
        ])

        # Apply pipeline to training data
        processed = pipeline.fit_transform(train)

        # Split into training and test sets
        x, y, x_test = FeatureEngineer.split_df(processed)
        x.to_csv(X_PATH, index=False)
        y.to_csv(Y_PATH, index=False)
        x_test.to_csv(X_TEST_PATH, index=False)
    else:
        # Load preprocessed data from disk
        x = pd.read_csv(X_PATH)
        y = pd.read_csv(Y_PATH)
        x_test = pd.read_csv(X_TEST_PATH)

    return x, y, x_test


def train_and_evaluate(run, x, y, x_test, spaces):
    trainer = ModelTrainer(
        model_type=MODEL_TYPE,
        metadata=MODEL_METADATA,
        spaces=spaces,
        neptune_run=run
    )
    trainer.train(x, y)
    rmse = trainer.evaluate_best_model(x, y)
    trainer.predict(x_test)
    trainer.save(MODEL_SAVE_PATH)
    return trainer, rmse


def log_to_neptune(run, trainer, rmse):
    # Log configuration parameters
    run["config/MODEL_TYPE"] = MODEL_TYPE
    run["eval/rmse"] = rmse
    run["config/LOG_TARGET"] = LOG_TARGET
    run["config/FORM_PREP_DATA"] = FORM_PREP_DATA
    run["config/USE_OPTUNA"] = USE_OPTUNA
    run["config/OPTUNA_SPACES"] = list(OPTUNA_SPACES.keys()) if OPTUNA_SPACES else None
    run["config/HYPEROPT_SPACES"] = list(HYPEROPT_SPACES.keys()) if HYPEROPT_SPACES else None

    # Log model parameters
    try:
        if MODEL_TYPE == "LSTM":
            config = trainer.model.get_config()
            run["parameters"] = str(config) if config is not None else "N/A"
        else:
            params = getattr(trainer.model, "get_params", lambda: None)()
            run["parameters"] = params if params is not None else "N/A"
    except Exception as e:
        run["parameters"] = f"Failed to extract parameters: {str(e)}"

    if hasattr(trainer, "best_params") and trainer.best_params:
        run["parameters/best"] = trainer.best_params

    # Upload model and DVC artifacts
    run["artifacts/model"].upload(str(MODEL_SAVE_PATH))
    if os.path.exists(".dvc/config"):
        run["dvc/config"].upload(".dvc/config")
    if os.path.exists("params.yaml"):
        run["dvc/params"].upload("params.yaml")

    # Log library versions
    run["env/xgboost_version"] = xgboost.__version__
    run["env/lightgbm_version"] = lightgbm.__version__
    run["env/sklearn_version"] = sklearn.__version__
    run["env/pandas_version"] = pd.__version__


def run_pipeline(NEPTUNE_API_TOKEN: str):
    logger.info("Sales Prediction Pipeline started")
    run = initialize_neptune_run(NEPTUNE_API_TOKEN)

    x, y, x_test = prepare_data(run)
    if LOG_TARGET:
        y = DataPreprocessor.log_transform(y)

    spaces = OPTUNA_SPACES if USE_OPTUNA else HYPEROPT_SPACES
    trainer, rmse = train_and_evaluate(run, x, y, x_test, spaces)
    log_to_neptune(run, trainer, rmse)
    run.stop()
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    neptune_token = get_neptune_token()
    run_pipeline(neptune_token)
