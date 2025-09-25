import numpy as np
import pandas as pd


class FeatureImportanceEvaluator:
    def __init__(self, model):
        self.model = model

    def get_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_.ravel())
        else:
            raise AttributeError(
                f"Model of type {type(self.model).__name__} does not support feature importance extraction.")

        return pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
