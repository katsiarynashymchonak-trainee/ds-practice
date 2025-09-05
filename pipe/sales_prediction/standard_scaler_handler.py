import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class StandardScalerHandler:
    scaler = StandardScaler()

    @staticmethod
    def scale_train(x_train: pd.DataFrame) -> pd.DataFrame:
        logger.info("🔧 Обучаем StandardScaler на тренировочных данных...")
        StandardScalerHandler.scaler.fit(x_train)
        scaled_train = StandardScalerHandler.scaler.transform(x_train)
        return pd.DataFrame(scaled_train, columns=x_train.columns, index=x_train.index)

    @staticmethod
    def scale_test(x_test: pd.DataFrame) -> pd.DataFrame:
        logger.info("📐 Применяем обученный scaler к тестовым данным...")
        scaled_test = StandardScalerHandler.scaler.transform(x_test)
        return pd.DataFrame(scaled_test, columns=x_test.columns, index=x_test.index)
