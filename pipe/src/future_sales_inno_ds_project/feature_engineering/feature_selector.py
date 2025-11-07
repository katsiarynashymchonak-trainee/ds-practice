# feature_selector.py

import logging
import time

import dill
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..config import RF_BASE_PATH, TRAIN_RFR_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureSelector:
    def __init__(self, model=None, random_state=42, verbose=True, importance_threshold=0.01):
        self.model = model or RandomForestRegressor(n_jobs=-1, max_depth=6, n_estimators=600)
        self.random_state = random_state
        self.verbose = verbose
        self.importance_threshold = importance_threshold
        self.selected_features_importance = []
        self.shap_values = None
        self.shap_importance = None

    def evaluate_feature_importance(self, x: pd.DataFrame, importances: np.ndarray) -> pd.DataFrame:
        importance_df = pd.DataFrame({
            'feature': x.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        self.selected_features_importance = importance_df[
            importance_df['importance'] >= self.importance_threshold
            ]['feature'].tolist()

        rejected = importance_df[
            importance_df['importance'] < self.importance_threshold
            ]['feature'].tolist()

        print(f"Importance-based selection: {len(self.selected_features_importance)} features out of {x.shape[1]}")
        print(f"Selected features: {self.selected_features_importance}")
        print(f"Rejected features: {rejected}")

        return importance_df

    # Fitting model and selecting features based on built-in importance
    def fit_importance(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        start = time.time()
        logger.info("Fitting model and evaluating feature importance...")

        if TRAIN_RFR_MODEL:
            logger.info("Training base RandomForestRegressor...")
            self.model.fit(x, y.values.ravel())
            logger.info(f"Saving model to {RF_BASE_PATH}")
            with open(RF_BASE_PATH, 'wb') as f:
                dill.dump(self.model, f)
        else:
            logger.info(f"Loading model from {RF_BASE_PATH}")
            with open(RF_BASE_PATH, 'rb') as f:
                self.model = dill.load(f)

        # Вызов нового метода
        importances = self.model.feature_importances_
        self.importance_df = self.evaluate_feature_importance(x, importances)

        logger.info(f"Feature importance evaluation completed in {time.time() - start:.2f}s")
        return x[self.selected_features_importance]

    # Transforming input data using selected features
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_importance:
            logger.info(f"Transforming input using {len(self.selected_features_importance)} selected features")
            return x[self.selected_features_importance]
        else:
            raise ValueError("Feature importance selection has not been fitted yet.")
