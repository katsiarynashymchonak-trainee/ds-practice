from keras import Input
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

from ..config import LSTM_INPUT_SHAPE

class ModelFactory:
    @staticmethod
    def create(model_type: str):
        if model_type == "XGBRegressor":
            return XGBRegressor()

        elif model_type == "LGBMRegressor":
            return LGBMRegressor()

        elif model_type == "RandomForestRegressor":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor()

        elif model_type == "RidgeRegression":
            return Ridge()

        elif model_type == "LSTM":
            return ModelFactory._build_lstm_model()

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _build_lstm_model():
        model = Sequential([
            Input(shape=LSTM_INPUT_SHAPE),
            LSTM(64),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
