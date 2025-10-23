# data_loader.py

import logging
import pandas as pd
from ..config import (
    TRAIN_PATH, TEST_PATH, ITEMS_PATH, SUB_PATH, CATS_PATH, SHOPS_PATH
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataLoader:
    def __init__(self):
        self.train_path = TRAIN_PATH
        self.test_path = TEST_PATH
        self.items_path = ITEMS_PATH
        self.categories_path = CATS_PATH
        self.shops_path = SHOPS_PATH
        self.sub_path = SUB_PATH

    def load_data(self):
        train = self.load_train()
        items = self.load_items()
        categories = self.load_categories()
        shops = self.load_shops()
        test = self.load_test(train["date_block_num"].max() + 1)

        return train, items, categories, shops, test

    def load_train(self):
        logger.info("Loading training data...")
        train = pd.read_csv(self.train_path)

        # Dates parsing
        train['date'] = pd.to_datetime(train['date'], errors='coerce', format='%d.%m.%Y')
        train["day"] = train["date"].dt.month
        train["month"] = train["date"].dt.month
        train["year"] = train["date"].dt.year
        train.drop(columns=["date"], inplace=True)

        # Valid price
        train = train[train["item_price"] > 0]
        return train

    def load_test(self, next_month_num):
        logger.info("Loading test data...")
        test = pd.read_csv(self.test_path)

        # Init test period
        test['date_block_num'] = next_month_num

        return test

    def load_items(self):
        logger.info("Loading item metadata...")
        items = pd.read_csv(self.items_path)
        return items

    def load_categories(self):
        logger.info("Loading categories metadata...")
        cats = pd.read_csv(self.categories_path)
        return cats

    def load_shops(self):
        logger.info("Loading shops metadata...")
        shops = pd.read_csv(self.shops_path)
        return shops

    def load_submission_file(self):
        logger.info("Loading submission file...")
        sub = pd.read_csv(self.sub_path)
        return sub
