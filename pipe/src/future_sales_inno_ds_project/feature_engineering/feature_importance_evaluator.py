import logging
import numpy as np
import pandas as pd
import neptune

logger = logging.getLogger(__name__)


class FeatureImportanceEvaluator:
    def __init__(self, model, neptune_run=None):
        self.model = model
        self.neptune_run = neptune_run

    def get_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            logger.info("Extracted feature importances from 'feature_importances_'")
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_.ravel())
            logger.info("Extracted feature importances from 'coef_'")
        else:
            msg = f"Model of type {type(self.model).__name__} does not support feature importance extraction."
            logger.error(msg)
            raise AttributeError(msg)

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        # Log to console
        logger.info("Top 10 features by importance:")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

        # Log to Neptune if available
        if self.neptune_run:
            self.neptune_run["feature_importance/table"] = neptune.types.File.as_html(importance_df)
            for i, row in importance_df.iterrows():
                self.neptune_run[f"feature_importance/{row['feature']}"] = row['importance']

        return importance_df
