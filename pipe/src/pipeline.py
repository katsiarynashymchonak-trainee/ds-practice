import logging

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

from config import (
    MODEL_SAVE_PATH, MODEL_METADATA, OPTUNA_SPACES,
    X_PATH, X_TEST_PATH, Y_PATH, FORM_PREP_DATA,
)
from pipe.src.data_preparation.data_loader import DataLoader
from pipe.src.data_preparation.data_preprocessor import DataPreprocessor
from pipe.src.feature_engineering.feature_engineer import FeatureEngineer
from pipe.src.modeling.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_pipeline():
    logger.info("Sales Prediction Pipeline")
    if FORM_PREP_DATA:
        # Loading data
        loader = DataLoader()
        train, items, categories, shops, test = loader.load_data()

        # Building pipeline
        pipeline = Pipeline([
            ("aggregate_data", FunctionTransformer(DataPreprocessor.aggregate_data)),
            ("create_full_train", FunctionTransformer(FeatureEngineer.create_full_train)),
            ("remove_shops", FunctionTransformer(FeatureEngineer.remove_shops_from_train, kw_args={"test": test})),
            ("remove_zero_sales", FunctionTransformer(DataPreprocessor.remove_zero_sales)),
            ("concat_data", FunctionTransformer(FeatureEngineer.concat_train_test, kw_args={"test": test})),
            ("replace_shop_ids", FunctionTransformer(DataPreprocessor.replace_shop_ids)),
            ("add_features",
             FunctionTransformer(FeatureEngineer.add_features,
                                 kw_args={"items": items, "categories": categories, "shops": shops})),
            ("cast_types", FunctionTransformer(DataPreprocessor.cast_types))
        ])

        processed = pipeline.fit_transform(train)

        # Train/test split
        x, y, x_test = FeatureEngineer.split_df(processed)
        x.to_csv(X_PATH, index=False)
        y.to_csv(Y_PATH, index=False)
        x_test.to_csv(X_TEST_PATH, index=False)
    else:
        x = pd.read_csv(X_PATH)
        y = pd.read_csv(Y_PATH)
        x_test = pd.read_csv(X_TEST_PATH)

    # Modeling
    trainer = ModelTrainer(
        models=(XGBRegressor(), ),
        # lgb.LGBMRegressor(verbose=-1), Ridge(), RandomForestRegressor(verbose=-1)
        metadata=MODEL_METADATA,
        param_spaces=OPTUNA_SPACES
    )
    trainer.train(x, y)
    trainer.evaluate_best_model(x, y)
    trainer.predict(x_test)

    # Saving best model
    trainer.save(MODEL_SAVE_PATH)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
