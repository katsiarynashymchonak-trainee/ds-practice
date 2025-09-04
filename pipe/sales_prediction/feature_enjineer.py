# feature_engineer.py
import logging
from itertools import product

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureEngineer:
    @staticmethod
    def create_full_train(data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating full grid of shop_id × item_id × date_block_num...")

        # Create full matrix of all combinations for each month
        grids = [
            np.array(list(product([block], shops, items)), dtype='int32')
            for block, group in data.groupby('date_block_num')
            for shops in [group['shop_id'].unique()]
            for items in [group['item_id'].unique()]
        ]
        grid_df = pd.DataFrame(
            np.vstack(grids),
            columns=['date_block_num', 'shop_id', 'item_id']
        )

        # Merge with original data to fill in missing combinations
        merged = pd.merge(grid_df, data, on=['date_block_num', 'shop_id', 'item_id'], how='left')

        # Fill missing sales with 0 and clip extreme values
        merged['item_cnt_month'] = (
            merged['item_cnt_month']
            .fillna(0)
            .clip(0, 600)
            .astype(np.int32)
        )
        return merged

    @staticmethod
    def concat_train_test(train, test):
        logger.info("Concatenating train and test...")

        # Combine train and test datasets for feature generation
        cols_to_concat = ['date_block_num', 'shop_id', 'item_id']
        target_col = ['item_cnt_month']

        df = pd.concat([train[cols_to_concat + target_col],
                        test[cols_to_concat]],
                       ignore_index=True,
                       sort=False)

        return df

    @staticmethod
    def add_features(data: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding features to dataset...")
        df = data.copy()

        # Merge item categories
        df = pd.merge(df, items[['item_id', 'item_category_id']], on='item_id', how='left')

        # Sort for consistent lag calculation
        df.sort_values(['date_block_num', 'shop_id', 'item_id'], inplace=True)

        # Add lag features for previous months
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}_month'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag).fillna(0)

        # Extract year and month from date_block_num
        df['year'] = (df['date_block_num'] // 12) + 2013
        df['month'] = (df['date_block_num'] % 12) + 1

        # Average item sales in previous month
        df['avg_item_cnt_prev_month'] = (
            df.groupby('item_id')['item_cnt_month']
            .shift(1)
            .fillna(0)
        )

        # Average shop sales in previous month
        df['avg_shop_cnt_prev_month'] = (
            df.groupby('shop_id')['item_cnt_month']
            .shift(1)
            .fillna(0)
        )

        for num in sorted(df["date_block_num"].unique()):
            filtered_data = df[df["date_block_num"] <= num]

            # First month when item was sold
            df.loc[df["date_block_num"] == num, 'item_first_month'] = filtered_data.groupby('item_id')[
                'date_block_num'].transform('min')
            # First month when shop made a sale
            df.loc[df["date_block_num"] == num, 'shop_first_month'] = filtered_data.groupby('shop_id')[
                'date_block_num'].transform('min')

        # Calculate item and shop age in months
        df['item_age_months'] = df['date_block_num'] - df['item_first_month']
        df['shop_age_months'] = df['date_block_num'] - df['shop_first_month']

        return df

    @staticmethod
    def split_df(data: pd.DataFrame):
        logger.info("Splitting data into train and test sets...")
        df = data.copy()

        # Separate final test and train sets
        final_test_features = df[df['date_block_num'] == 34].copy()
        final_train_features = df[df['date_block_num'] <= 33].copy()

        # Remove early months to avoid cold-start bias
        min_train_month = 12
        final_train_features = final_train_features[final_train_features['date_block_num'] >= min_train_month].copy()

        # Select feature columns
        feature_cols = [
            'lag_1_month', 'lag_2_month', 'lag_3_month', 'lag_6_month', 'lag_12_month',
            'avg_item_cnt_prev_month', 'avg_shop_cnt_prev_month', 'month', 'year',
            'item_age_months', 'shop_age_months',
            'item_category_id'
        ]

        # Split into X and y for training, and X_test for prediction
        x = final_train_features[feature_cols]
        y = final_train_features['item_cnt_month']
        x_test = final_test_features[feature_cols]

        return x, y, x_test
