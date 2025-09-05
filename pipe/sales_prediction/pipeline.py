import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

from config import (
    MODEL_SAVE_PATH, RIDGE_PARAMS, RF_PARAMS,
    XGB_PARAMS, LGB_PARAMS, MODEL_METADATA
)
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_enjineer import FeatureEngineer
from model_trainer import ModelTrainer
from utills.logger import setup_logger

logger = setup_logger("pipeline")


def run_pipeline():
    logger.info("Sales Prediction Pipeline")

    # DataLoader instance
    loader = DataLoader()

    # Loading data
    train = loader.load_train()
    items = loader.load_items()
    test = loader.load_test(train["date_block_num"].max() + 1)

    # Building pipeline
    pipeline = Pipeline([
        ("aggregate_data", FunctionTransformer(DataPreprocessor.aggregate_data)),
        ("create_full_train", FunctionTransformer(FeatureEngineer.create_full_train)),
        ("concat_data", FunctionTransformer(FeatureEngineer.concat_train_test, kw_args={"test": test})),
        ("replace_shop_ids", FunctionTransformer(DataPreprocessor.replace_shop_ids)),
        ("add_features", FunctionTransformer(FeatureEngineer.add_features, kw_args={"items": items})),
        ("cast_types", FunctionTransformer(DataPreprocessor.cast_types))
    ])

    processed = pipeline.fit_transform(train)

    # Train/test split
    x, y, x_test = FeatureEngineer.split_df(processed)

    # Models initialization
    xgb_model = XGBRegressor(**XGB_PARAMS)
    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS, callbacks=[lgb.early_stopping(stopping_rounds=100)])
    ridge_model = Ridge(**RIDGE_PARAMS)
    rf_model = RandomForestRegressor(**RF_PARAMS)
    models = (xgb_model, lgb_model, ridge_model, rf_model)

    # Modeling
    trainer = ModelTrainer(models=models, metadata=MODEL_METADATA)
    trainer.train(x, y, x_test)
    trainer.predict(x_test)
    # Save best model
    trainer.save(MODEL_SAVE_PATH)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
