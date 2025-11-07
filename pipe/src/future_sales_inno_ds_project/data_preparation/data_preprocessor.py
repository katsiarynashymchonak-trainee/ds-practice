# data_preprocessor.py

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataPreprocessor:

    # Replacing duplicate shop IDs with unified values
    @staticmethod
    def replace_shop_ids(df: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logger.info("Replacing shop IDs...")
        data = df.copy()

        shop_id_map = {
            0: 57,
            1: 58,
            10: 11
        }

        data["shop_id"] = data["shop_id"].replace(shop_id_map)
        logger.info(f"Shop ID replacement completed in {time.time() - start:.2f} seconds")
        return data

    # Aggregating daily sales into monthly totals per shop and item
    @staticmethod
    def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logger.info("Aggregating monthly sales...")
        data = df.copy()

        # Filtering out extreme values in price and quantity
        data = data[data.item_price < 100000]
        data = data[data.item_cnt_day < 1001]

        # Summing daily sales by month, shop, and item
        data_grouped = data.groupby(
            ["date_block_num", "shop_id", "item_id"],
            as_index=False
        ).agg({"item_cnt_day": "sum"})

        # Renaming column and clipping sales to [0, 20] range
        data_grouped.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
        data_grouped['item_cnt_month'] = data_grouped['item_cnt_month'].clip(0, 20)

        logger.info(f"Monthly aggregation completed in {time.time() - start:.2f} seconds")
        return data_grouped

    # Removing zero-sales rows from training data except for the final month
    @staticmethod
    def remove_zero_sales(df: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logger.info("Removing zero-sales rows from train...")
        data = df.copy()

        # Keeping rows with sales or from the last month
        data = data[(data['item_cnt_month'] > 0) | (data['date_block_num'] == data['date_block_num'].max())]

        logger.info(f"Zero-sales removal completed in {time.time() - start:.2f} seconds")
        return data

    # Casting numeric columns to compact integer types to reduce memory usage
    @staticmethod
    def cast_types(df: pd.DataFrame, cols=None):
        start = time.time()
        logger.info("Casting integer types based on value ranges...")
        data = df.copy()
        memory_before = data.memory_usage(deep=True).sum()

        # Selecting columns to cast
        if cols is None:
            cols = [col for col in data.columns if col != "item_cnt_month"]

        # Applying appropriate integer type based on min/max values
        for col in cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                max_val = data[col].max()
                min_val = data[col].min()

                if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                    data[col] = data[col].astype("int8")
                    logger.debug(f"{col}: cast to int8")
                elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                    data[col] = data[col].astype("int16")
                    logger.debug(f"{col}: cast to int16")
                else:
                    data[col] = data[col].astype("int32")
                    logger.debug(f"{col}: cast to int32")

        memory_after = data.memory_usage(deep=True).sum()
        print(f"Memory reduced from {memory_before:,} to {memory_after:,} bytes "
                    f"({memory_before / memory_after:.2f}× smaller)")
        logger.info(f"Type casting completed in {time.time() - start:.2f} seconds")

        return data

    @staticmethod
    def log_transform(y: pd.DataFrame) -> pd.DataFrame:
        y_copy = y.copy()
        y_log = np.log1p(y_copy)
        logger.info("Applied np.log1p to target variable.")
        return y_log

