# config.py
import os
from hyperopt import hp

# Получаем базовую директорию от текущего файла
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Main paths
TRAIN_PATH = os.path.join(BASE_DIR, "data/raw/sales_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data/raw/test.csv")
ITEMS_PATH = os.path.join(BASE_DIR, "data/raw/items.csv")
CATS_PATH = os.path.join(BASE_DIR, "data/raw/item_categories.csv")
SHOPS_PATH = os.path.join(BASE_DIR, "data/raw/shops.csv")
SUB_PATH = os.path.join(BASE_DIR, "data/raw/sample_submission.csv")
PRED_PATH = os.path.join(BASE_DIR, "data/processed/sample_submission.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/sales_model.pkl")

# Processed data files
X_PATH = os.path.join(BASE_DIR, "data/processed/x.csv")
X_TEST_PATH = os.path.join(BASE_DIR, "data/processed/x_test.csv")
Y_PATH = os.path.join(BASE_DIR, "data/processed/y.csv")

# Labels-value pairs
LE_CITY_PATH = os.path.join(BASE_DIR, "data/labels/shop_city_labels.csv")
LE_GROUPS_PATH = os.path.join(BASE_DIR, "data/labels/group_name_labels.csv")
LE_ITEMF4_PATH = os.path.join(BASE_DIR, "data/labels/item_first4_labels.csv")
LE_ITEMF6_PATH = os.path.join(BASE_DIR, "data/labels/item_first6_labels.csv")
LE_ITEMF11_PATH = os.path.join(BASE_DIR, "data/labels/item_first11_labels.csv")

# Remaining features for result model
FEATURE_COLS = [
            'lag_1_month', 'lag_2_month', 'lag_3_month',
            'avg_item_cnt_prev_month', 'avg_shop_cnt_prev_month', 'month',
            'item_age', 'shop_age', 'category_age',
            'item_category_id', 'shop_city',
            'category_cnt_lag1', 'category_cnt_all_shops_lag1',
            'group_cnt_lag1', 'group_cnt_all_shops_lag1',
            'city_cnt_lag1', 'year'
        ]

# To reform dataset
FORM_PREP_DATA = 0
# To use feature selection with rfr model
USE_FEATURE_SELECTION = 0
# Use random samples to tune hyperparams
SAMPLE_BEFORE_TUNING = 1

# To retrain base rfr model
RF_BASE_PATH = os.path.join(BASE_DIR, "models/rfr.pkl")
TRAIN_RFR_MODEL = 1

# Metadata
MODEL_METADATA = {
    "name": "Sales prediction model",
    "author": "Katsiaryna Shymchonak",
    "version": 1
}

HYPEROPT_SPACES = {
    "XGBRegressor": {
        "n_estimators": hp.choice("xgb_n_estimators", [100, 200, 300]),
        "max_depth": hp.choice("xgb_max_depth", [5, 6, 8]),
        "learning_rate": hp.uniform("xgb_learning_rate", 0.01, 0.3),
        "subsample": hp.uniform("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("xgb_colsample_bytree", 0.5, 1.0),  # Непрерывное распределение
        "reg_alpha": hp.uniform("xgb_reg_alpha", 0.0, 1.0),
        "reg_lambda": hp.uniform("xgb_reg_lambda", 0.0, 1.0)
    },

    "LGBMRegressor": {
        "boosting_type": hp.choice("lgb_boosting_type", ["gbdt", "dart"]),  # dart can help with weak splits
        "n_estimators": hp.choice("lgb_n_estimators", [100, 300, 500]),
        "learning_rate": hp.uniform("lgb_learning_rate", 0.01, 0.15),  # slightly lower upper bound

        # Avoid unrestricted depth and overly deep trees
        "max_depth": hp.choice("lgb_max_depth", [6, 9, 12]),

        # Moderate leaf count to avoid overfitting and unstable splits
        "num_leaves": hp.choice("lgb_num_leaves", [31, 64, 128]),

        # Lower min_data_in_leaf to allow splits on smaller groups
        "min_data_in_leaf": hp.choice("lgb_min_data_in_leaf", [10, 20, 50]),

        # Allow weak splits by lowering gain threshold
        "min_gain_to_split": hp.uniform("lgb_min_gain_to_split", 0.0, 0.2),

        # Keep feature and bagging fractions high for diversity
        "feature_fraction": hp.uniform("lgb_feature_fraction", 0.85, 1.0),
        "bagging_fraction": hp.uniform("lgb_bagging_fraction", 0.85, 1.0),
        "bagging_freq": hp.choice("lgb_bagging_freq", [1, 5]),

        # Stronger regularization to stabilize splits
        "lambda_l1": hp.uniform("lgb_lambda_l1", 0.1, 1.0),
        "lambda_l2": hp.uniform("lgb_lambda_l2", 0.1, 1.0),

        # Additional stability flags
        "force_col_wise": True,
        "verbosity": -1
    },

    "RandomForestRegressor": {
        "n_estimators": hp.choice("rf_n_estimators", [100, 200, 300]),
        "max_depth": hp.choice("rf_max_depth", [5, 10, 15]),
        "min_samples_split": hp.choice("rf_min_samples_split", [2, 5, 10]),
        "min_samples_leaf": hp.choice("rf_min_samples_leaf", [1, 2, 4]),
        "max_features": hp.choice("rf_max_features", ["sqrt", "log2", None])
    },

    "Ridge": {
        "alpha": hp.uniform("ridge_alpha", 0.1, 10.0),
        "fit_intercept": hp.choice("ridge_fit_intercept", [True, False]),
        "solver": hp.choice("ridge_solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg"])
    }
}


def xgb_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 1200, step=100),
        "max_depth": trial.suggest_int("xgb_max_depth", 4, 10),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 300),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
        "tree_method": "hist",  # ускоряет обучение на больших данных
        "objective": "reg:squarederror",
        "verbosity": 0
    }


def lgb_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("lgb_n_estimators", 300, 1200, step=100),
        "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("lgb_max_depth", 4, 12),
        "num_leaves": trial.suggest_int("lgb_num_leaves", 31, 256),
        "min_data_in_leaf": trial.suggest_int("lgb_min_data_in_leaf", 20, 100),
        "min_gain_to_split": trial.suggest_float("lgb_min_gain_to_split", 0.0, 0.5),
        "feature_fraction": trial.suggest_float("lgb_feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("lgb_bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("lgb_bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lgb_lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lgb_lambda_l2", 1e-8, 10.0, log=True),
        "force_col_wise": True,
        "verbosity": -1,
        "objective": "regression"
    }


def rf_param_space(trial):
    return {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("rf_max_depth", 4, 10),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2"]),
    }


def ridge_param_space(trial):
    return {
        "alpha": trial.suggest_float("ridge_alpha", 0.01, 100.0, log=True),
        "fit_intercept": trial.suggest_categorical("ridge_fit_intercept", [True, False]),
        "solver": trial.suggest_categorical("ridge_solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"])
    }


OPTUNA_SPACES = {
    "XGBRegressor": xgb_param_space,
    "LGBMRegressor": lgb_param_space,
    "RandomForestRegressor": rf_param_space,
    "Ridge": ridge_param_space
}
