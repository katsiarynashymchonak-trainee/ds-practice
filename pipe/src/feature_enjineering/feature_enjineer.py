# feature_engineer.py

import logging
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pipe.src.config import LE_CITY_PATH, LE_GROUPS_PATH, LE_ITEMF4_PATH, LE_ITEMF6_PATH, LE_ITEMF11_PATH, FEATURE_COLS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureEngineer:

    # Encoding categories and saving result to file
    @staticmethod
    def encode_category(data, filename='label_mapping.csv'):
        # Apply LabelEncoder
        le = LabelEncoder()
        encoded_values = le.fit_transform(data)

        # Save mapping into file
        mapping_df = pd.DataFrame({
            'Encoded Label': range(len(le.classes_)),
            'Original Value': le.classes_
        })
        mapping_df.to_csv(filename, index=False)
        logger.info(f'LE mapping saved to {filename}')

        return encoded_values

    # Removing shops that are not present in the test set
    @staticmethod
    def remove_shops_from_train(df: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logger.info("Removing shops not in test...")
        data = df.copy()
        data = data[data['shop_id'].isin(test.shop_id.unique())]
        logger.info(f"Shop filtering completed in {time.time() - start:.2f}s")
        return data

    # Generating full grid of shop_id × item_id × date_block_num combinations
    @staticmethod
    def create_full_train(df: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logger.info("Generating full training grid...")
        data = df.copy()

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

        merged = pd.merge(grid_df, data, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        merged['item_cnt_month'] = merged['item_cnt_month'].fillna(0).astype(np.int32)
        logger.info(f"Grid creation completed in {time.time() - start:.2f}s")
        logger.info(f"Shape after merge: {merged.shape}")
        return merged

    # Concatenating train and test sets for unified feature generation
    @staticmethod
    def concat_train_test(train, test):
        start = time.time()
        logger.info("Concatenating train and test...")
        cols_to_concat = ['date_block_num', 'shop_id', 'item_id']
        target_col = ['item_cnt_month']

        df = pd.concat([train[cols_to_concat + target_col], test[cols_to_concat]],
                       ignore_index=True, sort=False)
        logger.info(f"Concatenation completed in {time.time() - start:.2f}s")
        logger.info(f"Shape after concat: {df.shape}")
        return df

    # Adding aggregated lag feature based on specified grouping
    @staticmethod
    def agg_cnt_col_lagged(data, merging_cols, new_col, aggregation, lag=1, fillna_value=0):
        start = time.time()
        logger.info(f"Adding aggregated lag feature: {new_col}")
        temp = data.copy()
        temp['date_block_num'] += lag
        temp = temp.groupby(merging_cols).agg(aggregation).reset_index()
        temp.columns = merging_cols + [new_col]

        data = pd.merge(data, temp, on=merging_cols, how='left')
        data[new_col] = data[new_col].fillna(fillna_value)
        logger.info(f"{new_col} added in {time.time() - start:.2f}s")
        return data

    # Extracting group_id
    @staticmethod
    def extract_groups(categories: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logger.info("Encoding category groups...")

        categories = categories.copy()

        categories['item_category_name'] = categories['item_category_name'].astype(str)
        categories['group_name'] = categories['item_category_name'].str.extract(r'(^[\w\s]*)')[0].str.strip()

        categories['group_id'] = FeatureEngineer.encode_category(categories.group_name.values, LE_GROUPS_PATH)

        logger.info(f"Category encoding completed in {time.time() - start:.2f}s")
        return categories

    # Processing shop names and extracting city
    @staticmethod
    def process_shops(shops: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logger.info("Processing shop names and cities...")
        shops = shops.copy()
        shops['shop_name'] = shops['shop_name'].str.lower().str.replace(r'[^\w\d\s]', '', regex=True)
        shops['shop_city'] = shops['shop_name'].str.split().str[0]
        shops.loc[shops['shop_id'].isin([12, 55]), 'shop_city'] = 'online'
        shops['shop_city'] = FeatureEngineer.encode_category(shops.shop_city.values, LE_CITY_PATH)
        logger.info(f"Shop processing completed in {time.time() - start:.2f}s")
        return shops

    # Cleaning item names and extracting encoded prefixes
    @staticmethod
    def process_items(items: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logger.info("Processing item names...")
        items = items.copy()
        items['item_name'] = items['item_name'].str.lower().str.replace('.', '', regex=False)
        for pattern in [r'[^\w\d\s\.]', r'\bthe\b', r'\bin\b', r'\bis\b', r'\bfor\b', r'\bof\b',
                        r'\bon\b', r'\band\b', r'\bto\b', r'\bwith\b', r'\byo\b']:
            items['item_name'] = items['item_name'].str.replace(pattern, ' ', regex=True)
        items['item_name'] = items['item_name'].str.replace(r'\b.\b', ' ', regex=True)

        items['item_name_no_space'] = items['item_name'].str.replace(' ', '', regex=False)
        items['item_name_first4'] = items['item_name_no_space'].str[:4]
        items['item_name_first6'] = items['item_name_no_space'].str[:6]
        items['item_name_first11'] = items['item_name_no_space'].str[:11]

        items['item_name_first4'] = FeatureEngineer.encode_category(items['item_name_first4'].values, LE_ITEMF4_PATH)
        items['item_name_first6'] = FeatureEngineer.encode_category(items['item_name_first6'].values, LE_ITEMF6_PATH)
        items['item_name_first11'] = FeatureEngineer.encode_category(items['item_name_first11'].values, LE_ITEMF11_PATH)

        logger.info(f"Item processing completed in {time.time() - start:.2f}s")
        return items

    # Merging metadata into main dataframe
    @staticmethod
    def merge_metadata(df, items, categories, shops):
        start = time.time()
        logger.info("Merging metadata...")
        df = pd.merge(df, shops[['shop_id', 'shop_city']], on="shop_id", how="left")
        df = pd.merge(df, items[['item_id', 'item_category_id',
                                 'item_name_first4', 'item_name_first6', 'item_name_first11']],
                      on='item_id', how='left')
        df = pd.merge(df, categories[["item_category_id", "group_id"]],
                      on='item_category_id', how='left')
        logger.info(f"Metadata merge completed in {time.time() - start:.2f}s")
        logger.info(f"Shape after merge: {df.shape}")
        return df

    # Adding lag features for previous months
    @staticmethod
    def add_lag_features(df):
        start = time.time()
        logger.info("Adding lag features...")
        df = df.sort_values(['date_block_num', 'shop_id', 'item_id'])
        for lag in [1, 2, 3]:
            df[f'lag_{lag}_month'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag).fillna(0)
        logger.info(f"Lag features added in {time.time() - start:.2f}s")
        return df

    # Adding temporal features and age indicators
    @staticmethod
    def add_temporal_features(df):
        start = time.time()
        logger.info("Adding temporal features...")
        df['year'] = (df['date_block_num'] // 12) + 2013
        df['month'] = (df['date_block_num'] % 12) + 1

        df['avg_item_cnt_prev_month'] = df.groupby('item_id')['item_cnt_month'].shift(1).fillna(0)
        df['avg_shop_cnt_prev_month'] = df.groupby('shop_id')['item_cnt_month'].shift(1).fillna(0)

        for num in sorted(df["date_block_num"].unique()):
            filtered_data = df[df["date_block_num"] <= num]
            df.loc[df["date_block_num"] == num, 'item_first_month'] = filtered_data.groupby('item_id')[
                'date_block_num'].transform('min')
            df.loc[df["date_block_num"] == num, 'shop_first_month'] = filtered_data.groupby('shop_id')[
                'date_block_num'].transform('min')

        df['item_age'] = df['date_block_num'] - df['item_first_month']
        df['shop_age'] = df['date_block_num'] - df['shop_first_month']
        df['category_age'] = df['date_block_num'] - df.groupby('item_category_id')['date_block_num'].transform('min')
        df['group_age'] = df['date_block_num'] - df.groupby('group_id')['date_block_num'].transform('min')
        logger.info(f"Temporal features added in {time.time() - start:.2f}s")
        return df

    # Adding all engineered features to the dataset
    @staticmethod
    def add_features(data: pd.DataFrame, items: pd.DataFrame,
                     categories: pd.DataFrame, shops: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        logger.info("Starting full feature engineering...")

        # Encoding categories
        categories = FeatureEngineer.extract_groups(categories)

        # Processing shops
        shops = FeatureEngineer.process_shops(shops)

        # Processing items
        items = FeatureEngineer.process_items(items)

        # Merging metadata
        df = FeatureEngineer.merge_metadata(data, items, categories, shops)

        # Adding lag features
        df = FeatureEngineer.add_lag_features(df)

        # Adding temporal features
        df = FeatureEngineer.add_temporal_features(df)

        # Adding aggregated lag features
        df = FeatureEngineer.agg_cnt_col_lagged(df, ['date_block_num', 'item_category_id', 'shop_id'],
                                                'category_cnt_lag1', {'item_cnt_month': 'mean'})
        df = FeatureEngineer.agg_cnt_col_lagged(df, ['date_block_num', 'item_category_id', 'shop_id'],
                                                'category_cnt_median_lag1', {'item_cnt_month': 'median'})
        df = FeatureEngineer.agg_cnt_col_lagged(df, ['date_block_num', 'item_category_id'],
                                                'category_cnt_all_shops_lag1', {'item_cnt_month': 'mean'})
        df = FeatureEngineer.agg_cnt_col_lagged(df, ['date_block_num', 'item_category_id'],
                                                'category_cnt_all_shops_median_lag1', {'item_cnt_month': 'median'})
        df = FeatureEngineer.agg_cnt_col_lagged(df, ['date_block_num', 'group_id', 'shop_id'],
                                                'group_cnt_lag1', {'item_cnt_month': 'mean'})
        df = FeatureEngineer.agg_cnt_col_lagged(df, ['date_block_num', 'group_id'],
                                                'group_cnt_all_shops_lag1', {'item_cnt_month': 'mean'})
        df = FeatureEngineer.agg_cnt_col_lagged(df, ['date_block_num', 'shop_city'],
                                                'city_cnt_lag1', {'item_cnt_month': 'mean'})

        logger.info(f"Feature engineering completed in {time.time() - start:.2f}s")
        logger.info(f"Final feature set shape: {df.shape}")
        return df

    # Splitting dataset into training and test sets with feature selection
    @staticmethod
    def split_df(data: pd.DataFrame):
        start = time.time()
        logger.info("Splitting data into train and test sets...")

        df = data.copy()

        # Separating final test and train sets
        final_test_features = df[df['date_block_num'] == 34].copy()
        final_train_features = df[df['date_block_num'] <= 33].copy()

        # Removing early months to avoid cold-start bias
        min_train_month = 3
        final_train_features = final_train_features[final_train_features['date_block_num'] >= min_train_month].copy()

        # Selecting feature columns
        feature_cols = FEATURE_COLS

        # Splitting into X and y for training, and X_test for prediction
        x = final_train_features[feature_cols]
        y = final_train_features['item_cnt_month']
        x_test = final_test_features[feature_cols]

        logger.info(f"Train shape: X={x.shape}, y={y.shape}")
        logger.info(f"Test shape: X_test={x_test.shape}")
        logger.info(f"Split completed in {time.time() - start:.2f}s")

        return x, y, x_test
