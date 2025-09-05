import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class StandardScalerHandler:

    def __init__(self):
        self.scaler = StandardScaler()

    def scale_train(self, x_train: pd.DataFrame) -> pd.DataFrame:
        self.scaler.fit(x_train)
        scaled_train = self.scaler.transform(x_train)
        return pd.DataFrame(scaled_train, columns=x_train.columns, index=x_train.index)

    def scale_test(self, x_test: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying StandardScaler to test...")
        scaled_test = self.scaler.transform(x_test)
        return pd.DataFrame(scaled_test, columns=x_test.columns, index=x_test.index)
