import logging

import dill
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from pipe.sales_prediction.config import RF_BASE_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureSelector:
    def __init__(self, model=None, random_state=42, verbose=True, importance_threshold=0.01):
        self.model = model or RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=random_state)
        self.random_state = random_state
        self.verbose = verbose
        self.importance_threshold = importance_threshold
        self.selected_features_importance = []
        self.shap_values = None
        self.shap_importance = None

    def fit_importance(self, x, y):
        # logger.info("Training basic rfr model...")
        # self.model.fit(x, y.values.ravel())
        # logger.info(f"Saving model to {RF_BASE_PATH}")
        # with open(RF_BASE_PATH, 'wb') as f:
        #    dill.dump(self.model, f)

        logger.info(f"Loading model from {RF_BASE_PATH}")
        with open(RF_BASE_PATH, 'rb') as f:
            self.model = dill.load(f)

        # Оцениваем важность
        logger.info("Evaluating feature importance for it...")
        importances = self.model.feature_importances_
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

        return x[self.selected_features_importance]

    logging.getLogger(__name__)

    def fit_shap(self, x: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        # Sample for SHAP explanation
        x_sample = x.sample(min(20000, len(x)), random_state=42)

        # Create SHAP explainer
        explainer = shap.Explainer(self.model.predict, x_sample)

        self.shap_values = explainer(x_sample)

        # Compute mean absolute SHAP values
        shap_sum = np.abs(self.shap_values.values).mean(axis=0)

        self.shap_importance = pd.DataFrame({
            'feature': x.columns,
            'importance': shap_sum
        }).sort_values(by='importance', ascending=False)

        print("SHAP Feature Importance:")
        print(self.shap_importance)

        # Проверка типа importance перед сравнением
        importance_series = self.shap_importance['importance']

        # Сравнение с порогом
        selected_features = self.shap_importance[importance_series >= threshold]['feature'].tolist()
        rejected_features = self.shap_importance[importance_series < threshold]['feature'].tolist()

        print(f"Selected features (threshold={threshold}): {selected_features}")
        print(f"Selected features (threshold={threshold}): {rejected_features}")
        return x[selected_features]

    def transform(self, x):
        if self.selected_features_importance:
            return x[self.selected_features_importance]
        else:
            raise ValueError("Feature importance selection has not been fitted yet.")
