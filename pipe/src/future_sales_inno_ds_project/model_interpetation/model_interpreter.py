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
            self.log("Entering _explain_shap method...")
            self.log(f"Model type detected: {self.model_type}")
            self.log(f"Input shape: {X.shape}")

            # Ensure the SHAP output directory exists
            SHAP_DIR.mkdir(parents=True, exist_ok=True)
            self.log(f"Verified SHAP directory exists: {SHAP_DIR}")

            # Choose appropriate SHAP explainer
            if self.model_type in ["XGBREGRESSOR", "LGBMREGRESSOR"]:
                self.log("Using TreeExplainer for SHAP...")
                explainer = shap.TreeExplainer(self.model)
            else:
                self.log("Using generic SHAP Explainer with model.predict...")
                explainer = shap.Explainer(self.model.predict, X)

            # Compute SHAP values
            self.log("Computing SHAP values...")
            shap_values = explainer(X)
            self.log("SHAP values computed successfully.")

            # Generate and display summary plot
            self.log("Generating SHAP summary plot...")
            shap.summary_plot(shap_values, X, max_display=10, show=False)

            fig = plt.gcf()
            self.log(f"Saving SHAP plot to: {BEESWARM_PATH}")
            fig.savefig(BEESWARM_PATH, dpi=300, bbox_inches='tight')
            plt.close()

            if BEESWARM_PATH.exists():
                self.log(f"SHAP beeswarm plot saved successfully at {BEESWARM_PATH}")
            else:
                self.log(f"WARNING: SHAP beeswarm plot was not saved at {BEESWARM_PATH}")

            # Upload to Neptune if run is active
            if self.run:
                self.log("Uploading SHAP plot to Neptune...")
                safe_upload_image(self.run, "error_analysis/beeswarm", BEESWARM_PATH)
                self.log("Upload to Neptune completed.")
            else:
                self.log("Neptune run not active. Skipping upload.")

        except Exception as e:
            self.log(f"SHAP failed with error: {str(e)}")

    def _explain_lstm(self, X):
        try:
            self.log("Entering _explain_lstm method...")
            self.log(f"Input shape: {X.shape}")

            # Ensure the SHAP output directory exists
            SHAP_DIR.mkdir(parents=True, exist_ok=True)
            self.log(f"Verified SHAP directory exists: {SHAP_DIR}")

            # Prepare background and input samples
            background = X[:100].values.astype(np.float32).reshape((100, 1, X.shape[1]))
            inputs = X[:50].values.astype(np.float32).reshape((50, 1, X.shape[1]))
            self.log(f"Background shape: {background.shape}, Inputs shape: {inputs.shape}")

            if hasattr(self.model, 'predict'):
                self.log("Model has .predict method. Proceeding with KernelExplainer...")
                explainer = shap.KernelExplainer(lambda x: self.model.predict(x).flatten(), background)

                self.log("Computing SHAP values for LSTM...")
                shap_values = explainer.shap_values(inputs)

                if isinstance(shap_values, list):
                    self.log("SHAP values returned as list. Using first element.")
                    shap_values = shap_values[0]

                reshaped_inputs = inputs.reshape(inputs.shape[0], -1)
                self.log(f"Reshaped inputs for plotting: {reshaped_inputs.shape}")

                self.log("Generating SHAP summary plot for LSTM...")
                shap.summary_plot(shap_values, reshaped_inputs, show=False)

                fig = plt.gcf()
                self.log(f"Saving LSTM SHAP plot to: {SHAP_LSTM_PATH}")
                fig.savefig(SHAP_LSTM_PATH, dpi=300, bbox_inches='tight')
                plt.close()

                if SHAP_LSTM_PATH.exists():
                    self.log(f"LSTM SHAP summary plot saved successfully at {SHAP_LSTM_PATH}")
                else:
                    self.log(f"WARNING: LSTM SHAP plot was not saved at {SHAP_LSTM_PATH}")

                if self.run:
                    self.log("Uploading LSTM SHAP plot to Neptune...")
                    safe_upload_image(self.run, "error_analysis/lstm_summary", SHAP_LSTM_PATH)
                    self.log("Upload to Neptune completed.")
                else:
                    self.log("Neptune run not active. Skipping upload.")
            else:
                self.log("Model does not support .predict method. Skipping SHAP interpretation.")

        except Exception as e:
            self.log(f"LSTM SHAP failed with error: {str(e)}")

