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


def run_pipeline(NEPTUNE_API_TOKEN: str):
    logger.info("Sales Prediction Pipeline started")

    # Initialize Neptune experiment
    run = neptune.init_run(
        project="katsiaryna.shymchonak/Future-sales",
        api_token=NEPTUNE_API_TOKEN,
    )

    if FORM_PREP_DATA:
        # Load raw data
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

        # Split into train and test sets
        x, y, x_test = FeatureEngineer.split_df(processed)
        x.to_csv(X_PATH, index=False)
        y.to_csv(Y_PATH, index=False)
        x_test.to_csv(X_TEST_PATH, index=False)
    else:
        # Load preprocessed data from disk
        x = pd.read_csv(X_PATH)
        y = pd.read_csv(Y_PATH)
        x_test = pd.read_csv(X_TEST_PATH)

    if LOG_TARGET:
        y = DataPreprocessor.log_transform(y)
    spaces = OPTUNA_SPACES if USE_OPTUNA else HYPEROPT_SPACES

    # Train and evaluate model
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

    # Log model parameters to Neptune (robust across model types)
    try:
        if MODEL_TYPE == "LSTM":
            config = trainer.model.get_config()
            run["parameters"] = str(config) if config is not None else "N/A"
        else:
            params = getattr(trainer.model, "get_params", lambda: None)()
            run["parameters"] = params if params is not None else "N/A"
    except Exception as e:
        run["parameters"] = f"Failed to extract parameters: {str(e)}"

    run["eval/rmse"] = rmse
    run["model/type"] = MODEL_TYPE
    run["artifacts/model"].upload(str(MODEL_SAVE_PATH))

    # Log best parameters if available
    if hasattr(trainer, "best_params") and trainer.best_params:
        run["parameters/best"] = trainer.best_params

    # Log library versions
    run["env/xgboost_version"] = xgboost.__version__
    run["env/lightgbm_version"] = lightgbm.__version__
    run["env/sklearn_version"] = sklearn.__version__
    run["env/pandas_version"] = pd.__version__

    # Log DVC config and params if available
    if os.path.exists(".dvc/config"):
        run["dvc/config"].upload(".dvc/config")
    if os.path.exists("params.yaml"):
        run["dvc/params"].upload("params.yaml")

    # Stop Neptune run
    run.stop()
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    neptune_token = get_neptune_token()
    run_pipeline(neptune_token)
