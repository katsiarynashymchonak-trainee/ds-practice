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

        # Группировка по date_block_num и получение уникальных shop_id и item_id
        grids = [
            np.array(list(product([block], shops, items)), dtype='int32')
            for block, group in data.groupby('date_block_num')
            for shops in [group['shop_id'].unique()]
            for items in [group['item_id'].unique()]
        ]

        # Объединение всех блоков в единую матрицу
        grid_df = pd.DataFrame(
            np.vstack(grids),
            columns=['date_block_num', 'shop_id', 'item_id']
        )

        # Объединение с исходными данными
        merged = pd.merge(grid_df, data, on=['date_block_num', 'shop_id', 'item_id'], how='left')

        # Заполнение пропусков и ограничение значений
        merged['item_cnt_month'] = (
            merged['item_cnt_month']
            .fillna(0)
            .clip(0, 500)
            .astype(np.int32)
        )

        return merged

    @staticmethod
    def concat_data(train, test):
        logger.info("Concatenating train and test...")
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

        # Добавление категории товара
        df = pd.merge(df, items[['item_id', 'item_category_id']], on='item_id', how='left')

        # Сортировка
        df.sort_values(['date_block_num', 'shop_id', 'item_id'], inplace=True)

        # Создание лагов
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}_month'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag).fillna(0)

        # Добавление месяца и года
        df['year'] = (df['date_block_num'] // 12) + 2013
        df['month'] = (df['date_block_num'] % 12) + 1

        # Сортировка по item_id и времени
        df.sort_values(by=['item_id', 'date_block_num'], inplace=True)

        # Средние продажи товара в предыдущем месяце
        df['avg_item_cnt_prev_month'] = (
            df.groupby('item_id')['item_cnt_month']
            .shift(1)
            .fillna(0)
        )

        # Средние продажи магазина в предыдущем месяце
        df['avg_shop_cnt_prev_month'] = (
            df.groupby('shop_id')['item_cnt_month']
            .shift(1)
            .fillna(0)
        )

        for num in sorted(df["date_block_num"].unique()):
            filtered_data = df[df["date_block_num"] <= num]

            # Month of item's first sale
            df.loc[df["date_block_num"] == num, 'item_first_month'] = filtered_data.groupby('item_id')[
                'date_block_num'].transform('min')
            # Month of first sale in shop
            df.loc[df["date_block_num"] == num, 'shop_first_month'] = filtered_data.groupby('shop_id')[
                'date_block_num'].transform('min')

        # Возраст товара
        df['item_age_months'] = df['date_block_num'] - df['item_first_month']
        # Возраст магазина
        df['shop_age_months'] = df['date_block_num'] - df['shop_first_month']

        return df

    @staticmethod
    def split_df(data: pd.DataFrame):
        logger.info("Splitting data into train and test sets...")
        df = data.copy()

        final_test_features = df[df['date_block_num'] == 34].copy()
        final_train_features = df[df['date_block_num'] <= 33].copy()

        # Исключаем ранние записи
        min_train_month = 12
        final_train_features = final_train_features[final_train_features['date_block_num'] >= min_train_month].copy()

        feature_cols = [
            'lag_1_month', 'lag_2_month', 'lag_3_month', 'lag_6_month', 'lag_12_month',
            'avg_item_cnt_prev_month', 'avg_shop_cnt_prev_month', 'month', 'year',
            'item_age_months', 'shop_age_months',
            'item_category_id'
        ]

        # Train/Validation
        x_train = final_train_features[feature_cols]
        y_train = final_train_features['item_cnt_month']
        x_test = final_test_features[feature_cols]

        return x_train, y_train, x_test
