# data_preprocessor.py

import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataPreprocessor:

    # Replaces shop IDs to unify duplicates
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

    # Aggregates daily sales into monthly totals per shop and item
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

    # Converts selected columns to int32 to reduce memory usage
    @staticmethod
    def cast_types(df: pd.DataFrame, cols=None):
        logger.info("Converting types to int32")
        data = df.copy()
        memory_before_reduction = sum(data.memory_usage())

        if cols is None:
            cols = [col for col in data.columns if col != "item_cnt_month"]

        for col in cols:
            data[col] = data[col].astype("int32")

        memory_after_reduction = sum(data.memory_usage())
        logger.info(f"Memory usage became {memory_before_reduction / memory_after_reduction} times smaller")

        return data
