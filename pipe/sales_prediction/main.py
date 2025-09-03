from itertools import product

import dill
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBRegressor


def preprocess_train(df):
    train = df.copy()
    train = prepare_train(train)
    train = aggregate_data(train)
    train = create_full_train(train)
    return train


def prepare_train(df):
    print("Preparing data...")
    data = df.copy()

    data.drop_duplicates(inplace=True)  # обработка дубликатов

    # Удаление строк с отрицательной ценой товара
    data = data[data["item_price"] >= 0]

    # Приведение типов к int32
    int_columns = data.columns.drop("item_price")
    data[int_columns] = data[int_columns].astype("int32")

    return data


def replace_shop_ids(df):
    print("Replacing shop ids...")
    data = df.copy()

    shop_id_map = {
        0: 57,  # Якутск Орджоникидзе, 56
        1: 58,  # Якутск ТЦ "Центральный"
        10: 11  # Жуковский ул. Чкалова 39м²
    }
    # Замена shop_id
    data['shop_id'] = data['shop_id'].replace(shop_id_map)

    return data


def aggregate_data(df):
    print("Aggregating data...")
    data = df.copy()

    # Агрегация
    data_grouped = data.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).agg({
        'item_cnt_day': 'sum'  # получаем количество проданного товара за месяц
    })
    data_grouped.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)

    # Ограничим значения целевого признака
    data_grouped['item_cnt_month'] = data_grouped['item_cnt_month'].clip(0, 600)

    return data_grouped


def load_train():
    print("Loading train...")
    train = pd.read_csv("../data/sales_train.csv")
    # Парсинг дат
    train['date'] = pd.to_datetime(train['date'], errors='coerce', format='%d.%m.%Y')

    train["day"] = train["day"].apply(lambda x: x.day)
    train["month"] = train["date"].apply(lambda x: x.month)
    train["year"] = train["date"].apply(lambda x: x.year)

    # Удаляем date_time признак
    train = train.drop(columns=["date"])

    return train


def load_test(next_month_num):
    print("Loading test...")
    test = pd.read_csv("../data/test.csv")
    test['date_block_num'] = next_month_num  # 34 номер ноября 2015 года
    return test


def load_items():
    print("Loading items...")
    items = pd.read_csv("../data/items.csv")
    return items


def concat_data(train, test):
    print("Concatenating train and test...")
    cols_to_concat = ['date_block_num', 'shop_id', 'item_id']
    target_col = ['item_cnt_month']

    df = pd.concat([train[cols_to_concat + target_col],
                    test[cols_to_concat]],
                   ignore_index=True,
                   sort=False)
    return df


def create_full_train(data):
    print("Generating full grid of ids...")

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


def add_features(data, items):
    print("Adding features...")
    df = data.copy()
    # Добавление категории товара
    df = pd.merge(df, items[['item_id', 'item_category_id']], on='item_id', how='left')

    # Сортировка
    df.sort_values(['date_block_num', 'shop_id', 'item_id'], inplace=True)

    # Создание лагов
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}_month'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag).fillna(0)

    # # Добавление месяца
    df['year'] = (df['date_block_num'] // 12) + 2013
    df['month'] = (df['date_block_num'] % 12) + 1

    df = df.sort_values(by=['item_id', 'date_block_num'])

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

    # Месяц начала продаж товара
    df['item_first_month'] = df.groupby('item_id')['date_block_num'].transform('min')
    # Возраст товара
    df['item_age_months'] = df['date_block_num'] - df['item_first_month']

    # Месяц начала продаж в магазине
    df['shop_first_month'] = df.groupby('shop_id')['date_block_num'].transform('min')
    # Возраст магазина
    df['shop_age_months'] = df['date_block_num'] - df['shop_first_month']

    return df


def split_df(data):
    print(f"Preparing data for modeling...")
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


# Стандартизация признаков
def apply_standard_scaling(df):
    data = df.copy()
    cols = [col for col in df.columns if col not in ["item_cnt_month", "date_block_num"]]
    print("Scaling data...")
    scaler = StandardScaler()
    data[cols] = scaler.fit_transform(data[cols])
    return data


def apply_standard_scaling_cols(df, cols):
    data = df.copy()
    print(f"Scaling columns {cols}...")
    scaler = StandardScaler()
    data[cols] = scaler.fit_transform(data[cols])
    return data


def main():
    print("Sales Prediction Pipeline")

    train = load_train()
    items = load_items()
    test = load_test(train["date_block_num"].max() + 1)

    preprocessor = Pipeline([
        ("preprocess_train", FunctionTransformer(preprocess_train)),
        ("merge_data", FunctionTransformer(concat_data, kw_args={"test": test})),
        ("replace_shop_ids", FunctionTransformer(replace_shop_ids)),
        ("add_features", FunctionTransformer(add_features, kw_args={"items": items})),
        ("scale", FunctionTransformer(apply_standard_scaling))
    ])

    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'learning_rate': 0.02,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'n_jobs': -1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }

    lgb_params = {
        'objective': 'regression',
        'n_estimators': 300,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'n_jobs': -1,
        'seed': 42
    }

    models = (
        XGBRegressor(**xgb_params),
        lgb.LGBMRegressor(**lgb_params, callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)])
    )

    best_score = 0.0
    best_pipe = None

    processed = preprocessor.fit_transform(train)
    x, y, x_test = split_df(processed)
    print("Modeling...")
    for model in models:
        tscv = TimeSeriesSplit(n_splits=4)

        if np.any(x.isnull()) or np.any(x_test.isnull()):
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('model', model)
            ])
        else:
            pipe = Pipeline([
                ('model', model)
            ])

        scores = cross_val_score(pipe, x, y, cv=tscv, scoring="neg_root_mean_squared_error")
        print(f"RMSE по временной кросс-валидации модели {type(model).__name__}:"
              f" {scores.mean():.4f} ± {scores.std():.4f}")

        if scores.mean() > best_score:
            best_score = scores.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["regressor"]).__name__}, rmse: {best_score:.4f}')
    best_pipe.fit(x, y)
    with open('sales_pipe.pkl', 'wb') as file:
        dill.dump({
            "model": best_pipe,
            "metadata": {
                "name": "Sales prediction model",
                "author": "Ekaterina Shimchyonok",
                "version": 1,
                "type": type(best_pipe.named_steps["regressor"]).__name__,
                "rmse ": best_score
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
