import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error

from ..config import HISTOGRAM_PATH, DYNAMIC_VS_ERROR_PATH, BEESWARM_PATH, SHAP_LSTM_PATH
from ..feature_engineering.feature_importance_evaluator import FeatureImportanceEvaluator

logger = logging.getLogger(__name__)


class ModelInterpreter:
    def __init__(self, model, model_type, validator, neptune_run=None):
        self.model = model
        self.model_type = model_type.upper()
        self.validator = validator
        self.run = neptune_run
        self.fe_importance = FeatureImportanceEvaluator(model, neptune_run)

    def log(self, message):
        logger.info(message)
        if self.run:
            self.run["logs"].log(message)

    def analyze(self):
        df = self.validator.error_log.copy()
        self.log(f"Error log shape: {df.shape}")

        # Fold-wise RMSE
        rmse_by_fold = df.groupby("fold").apply(
            lambda g: mean_squared_error(g["y_true"], g["y_pred"], squared=False)
        )
        self.log(f"Fold-wise RMSE:\n{rmse_by_fold}")
        if self.run:
            for fold, val in rmse_by_fold.items():
                self.run[f"metrics/fold_{fold}_rmse"] = val

        # Global error distribution
        self._plot_error_distribution(df)
        self._plot_dynamic_vs_error(df)

    def _plot_error_distribution(self, df):
        plt.figure()
        plt.hist(df["error"], bins=30, color="skyblue", edgecolor="black")
        plt.title("Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(HISTOGRAM_PATH)
        self.log("Saved error histogram.")

        if self.run:
            self.run["error_analysis/histogram"].upload(str(HISTOGRAM_PATH))

    def _plot_dynamic_vs_error(self, df):
        plt.figure()
        plt.scatter(df["target_dynamic"], np.abs(df["error"]), alpha=0.5)
        plt.title("Target Dynamic vs Absolute Error")
        plt.xlabel("Target Dynamic")
        plt.ylabel("Abs Error")
        plt.tight_layout()
        plt.savefig(DYNAMIC_VS_ERROR_PATH)
        self.log("Saved dynamic vs error plot.")

        if self.run:
            self.run["error_analysis/dynamic_vs_error"].upload(str(DYNAMIC_VS_ERROR_PATH))

    def explain(self, X: pd.DataFrame):
        self.log(f"Starting interpretation for {self.model_type}...")

        try:
            importance_df = self.fe_importance.get_importance(X)
            self.log("Used native feature importance.")
        except AttributeError:
            self.log("Falling back to SHAP...")
            if self.model_type == "LSTM":
                self._explain_lstm(X)
            else:
                self._explain_shap(X)

    def _explain_shap(self, X):
        try:
            if self.model_type in ["XGBREGRESSOR", "LGBMREGRESSOR"]:
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.Explainer(self.model.predict, X)

            shap_values = explainer(X)
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            plt.tight_layout()
            plt.savefig(BEESWARM_PATH)
            self.log("Saved SHAP beeswarm plot.")

            if self.run:
                self.run["shap/beeswarm"].upload(str(BEESWARM_PATH))
        except Exception as e:
            self.log(f"SHAP failed: {str(e)}")

    def _explain_lstm(self, X):
        try:
            # Transform pandas DataFrame to numpy
            background = X[:100].values.astype(np.float32)
            inputs = X[:50].values.astype(np.float32)

            if hasattr(self.model, 'predict'):
                explainer = shap.KernelExplainer(self.model.predict, background)
                shap_values = explainer.shap_values(inputs)

                shap.summary_plot(shap_values, inputs, show=False)
                plt.tight_layout()
                plt.savefig(SHAP_LSTM_PATH)
                self.log("Saved LSTM SHAP summary.")

                if self.run:
                    self.run["shap/lstm_summary"].upload(str(SHAP_LSTM_PATH))
            else:
                self.log("Model does not support .predict method.")
        except Exception as e:
            self.log(f"LSTM SHAP failed: {str(e)}")
