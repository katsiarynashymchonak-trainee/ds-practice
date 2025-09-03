# data_preprocessor.py

import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataPreprocessor:
    @staticmethod
    def prepare_train(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preparing training data...")
        data = df.copy()

        data = data[data["item_price"] >= 0]

        int_columns = data.columns.drop("item_price")
        data[int_columns] = data[int_columns].astype("int32")

        return data

    @staticmethod
    def replace_shop_ids(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Replacing shop IDs...")
        data = df.copy()

        shop_id_map = {
            0: 57,
            1: 58,
            10: 11
        }

        data["shop_id"] = data["shop_id"].replace(shop_id_map)
        return data

    @staticmethod
    def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Aggregating monthly sales...")
        data = df.copy()

        data_grouped = data.groupby(
            ["date_block_num", "shop_id", "item_id"],
            as_index=False
        ).agg({"item_cnt_day": "sum"})

        data_grouped.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
        data_grouped["item_cnt_month"] = data_grouped["item_cnt_month"].clip(0, 600)

        return data_grouped

    @staticmethod
    def apply_standard_scaling(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying standard scaling to all numeric features...")
        data = df.copy()
        cols = [col for col in df.columns if col not in ["item_cnt_month", "date_block_num"]]

        scaler = StandardScaler()
        data[cols] = scaler.fit_transform(data[cols])

        return data
