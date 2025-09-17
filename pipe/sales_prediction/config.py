# config.py

# Main paths
TRAIN_PATH = "../data/raw/sales_train.csv"
TEST_PATH = "../data/raw/test.csv"
ITEMS_PATH = "../data/raw/items.csv"
MODEL_SAVE_PATH = "../models/sales_model.pkl"

# Metadata
MODEL_METADATA = {
    "name": "Sales prediction model",
    "author": "Katsiaryna Shymchonak",
    "version": 1
}

# XGBoost params
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 300,
    'learning_rate': 0.02,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'n_jobs': -1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}

# LightGBM params
LGB_PARAMS = {
    'objective': 'regression',
    'n_estimators': 300,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'n_jobs': -1,
    'verbose': -1
}

# Random forest params
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'n_jobs': -1,
}

# Ridge regression params
RIDGE_PARAMS = {
    'alpha': 1.0,
    'fit_intercept': True,
    'solver': 'auto',
}