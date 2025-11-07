import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..config import HISTOGRAM_PATH, DYNAMIC_VS_ERROR_PATH, BEESWARM_PATH, SHAP_LSTM_PATH, SHAP_DIR
from ..feature_engineering.feature_importance_evaluator import FeatureImportanceEvaluator
from ..utills.neptune_utils import safe_upload_image

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
            safe_upload_image(self.run, "error_analysis/histogram", HISTOGRAM_PATH)

    def _plot_dynamic_vs_error(self, df):
        plt.figure()
        plt.scatter(df["target_dynamic"], df["error"], alpha=0.5)
        plt.title("Target Dynamic vs Mean Error")
        plt.xlabel("Target Dynamic")
        plt.ylabel("Mean Error")
        plt.tight_layout()
        plt.savefig(DYNAMIC_VS_ERROR_PATH)
        self.log("Saved dynamic vs error plot.")

        if self.run:
            safe_upload_image(self.run, "error_analysis/dynamic_vs_error", DYNAMIC_VS_ERROR_PATH)

    def explain(self, X: pd.DataFrame):
        self.log(f"Starting interpretation for {self.model_type}...")

        # Try to get native feature importance (e.g., from tree-based models)
        try:
            importance_df = self.fe_importance.get_importance(X)
            self.log("Used native feature importance.")
            # Optional: visualize or log importance_df here if needed
        except Exception as e:
            importance_df = None
            self.log(f"Native feature importance unavailable: {str(e)}")

        # Always run SHAP interpretation depending on model type
        if self.model_type == "LSTM":
            self._explain_lstm(X)
        else:
            self._explain_shap(X)

    def _explain_shap(self, X):
        try:
            SHAP_DIR.mkdir(parents=True, exist_ok=True)

            if self.model_type in ["XGBREGRESSOR", "LGBMREGRESSOR"]:
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.Explainer(self.model.predict, X)

            shap_values = explainer(X)
            shap.summary_plot(shap_values, X, max_display=10, show=False)

            fig = plt.gcf()
            fig.savefig(BEESWARM_PATH, dpi=300, bbox_inches='tight')
            plt.close(fig)

            if BEESWARM_PATH.exists():
                self.log(f"SHAP plot saved at {BEESWARM_PATH}")
                if self.run:
                    safe_upload_image(self.run, "error_analysis/beeswarm", BEESWARM_PATH)
            else:
                self.log(f"SHAP plot not saved at {BEESWARM_PATH}")

        except Exception as e:
            self.log(f"SHAP failed: {str(e)}")

    def _explain_lstm(self, X):
        try:
            SHAP_DIR.mkdir(parents=True, exist_ok=True)

            background = X[:100].values.astype(np.float32).reshape((100, 1, X.shape[1]))
            inputs = X[:50].values.astype(np.float32).reshape((50, 1, X.shape[1]))

            if hasattr(self.model, 'predict'):
                explainer = shap.KernelExplainer(lambda x: self.model.predict(x).flatten(), background)
                shap_values = explainer.shap_values(inputs)

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                reshaped_inputs = inputs.reshape(inputs.shape[0], -1)
                shap.summary_plot(shap_values, reshaped_inputs, show=False)

                fig = plt.gcf()
                fig.savefig(SHAP_LSTM_PATH, dpi=300, bbox_inches='tight')
                plt.close(fig)

                if SHAP_LSTM_PATH.exists():
                    self.log(f"LSTM SHAP plot saved at {SHAP_LSTM_PATH}")
                    if self.run:
                        safe_upload_image(self.run, "error_analysis/lstm_summary", SHAP_LSTM_PATH)
                else:
                    self.log(f"LSTM SHAP plot not saved at {SHAP_LSTM_PATH}")
            else:
                self.log("Model does not support .predict method. SHAP skipped.")

        except Exception as e:
            self.log(f"LSTM SHAP failed: {str(e)}")
