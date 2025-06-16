import utils as utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Hyperparameters
TIME_STEPS = 20
HORIZON = 15
BATCH_SIZE = 512
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 64
GRAD_CLIP = 1.0

def add_time_features(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    data['minute'] = data['timestamp'].dt.minute
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek  

    data['minute_sin'] = np.sin(data['minute'] * (2 * np.pi / 60))
    data['minute_cos'] = np.cos(data['minute'] * (2 * np.pi / 60))

    data['hour_sin'] = np.sin(data['hour'] * (2 * np.pi / 24))
    data['hour_cos'] = np.cos(data['hour'] * (2 * np.pi / 24))

    data['day_of_week_sin'] = np.sin(data['day_of_week'] * (2 * np.pi / 7))
    data['day_of_week_cos'] = np.cos(data['day_of_week'] * (2 * np.pi / 7))

    data = data.drop(['minute', 'hour', 'day_of_week'], axis=1)
    return data

def compute_trend(df):
    close = df['close'].values
    n = len(close) - HORIZON
    trend = np.zeros(len(close), dtype=int)
    delta = close[HORIZON:] - close[:-HORIZON]
    trend[:n][delta > 0] = 1
    trend[:n][delta < 0] = 2
    # last HORIZON remain 0 (flat)
    return trend

def scale_data(train, test, cols, isTrend=True):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    target_column = 'close'
    if isTrend == False:
        y_train = y_scaler.fit_transform(train[target_column].values.reshape(-1, 1))
        y_test = y_scaler.transform(test[target_column].values.reshape(-1, 1))
    X_train = x_scaler.fit_transform(train[cols])
    X_test  = x_scaler.transform(test[cols])        

    return x_scaler, y_scaler, X_train, X_test, y_train if isTrend == False else None, y_test if isTrend == False else None

def create_sequences(x, y, time_steps=TIME_STEPS, horizon=HORIZON):
    Xs, ys = [], []
    max_i = len(x) - time_steps - horizon + 1
    for i in range(max_i):
        Xs.append(x[i:i+time_steps])
        ys.append(y[i + time_steps + horizon - 1])
    return np.array(Xs), np.array(ys)

def get_feature_cols(i):
    # --- Feature sets ---
    feature_cols_0 = ['open', 'high', 'low', 'volume','close']
    feature_cols_1 = ['open', 'high', 'low', 'volume', 'close','minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
    feature_cols_2 = ['close','ask1_price_trend_60', 'bid1_price_trend_60', 'bid1_size_n', 'bid2_size_n', 'bid3_size_n', 'bid4_size_n', 'bid5_size_n', 'ask1_size_n', 'ask2_size_n', 'ask3_size_n', 'ask4_size_n', 'ask5_size_n', 'log_return_bid1_price', 'log_return_bid2_price', 'log_return_ask1_price', 'log_return_ask2_price', 'buy_spread_trend_60', 'sell_spread_trend_60', 'wap_1_trend_60', 'wap_2_trend_60', 'buy_vwap_trend_60', 'sell_vwap_trend_60', 'volume_trend_60']
    feature_cols_3 = ['close','ask1_price_trend_60', 'bid1_price_trend_60', 'bid1_size_n', 'bid2_size_n', 'bid3_size_n', 'bid4_size_n', 'bid5_size_n', 'ask1_size_n', 'ask2_size_n', 'ask3_size_n', 'ask4_size_n', 'ask5_size_n', 'log_return_bid1_price', 'log_return_bid2_price', 'log_return_ask1_price', 'log_return_ask2_price', 'buy_spread_trend_60', 'sell_spread_trend_60', 'wap_1_trend_60', 'wap_2_trend_60', 'buy_vwap_trend_60', 'sell_vwap_trend_60', 'volume_trend_60', 'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
    feature_cols_4 = ['open', 'high', 'low', 'volume', 'close', 'ask1_price_trend_60', 'bid1_price_trend_60', 'bid1_size_n', 'bid2_size_n', 'bid3_size_n', 'bid4_size_n', 'bid5_size_n', 'ask1_size_n', 'ask2_size_n', 'ask3_size_n', 'ask4_size_n', 'ask5_size_n', 'log_return_bid1_price', 'log_return_bid2_price', 'log_return_ask1_price', 'log_return_ask2_price', 'buy_spread_trend_60', 'sell_spread_trend_60', 'wap_1_trend_60', 'wap_2_trend_60', 'buy_vwap_trend_60', 'sell_vwap_trend_60', 'volume_trend_60']
    feature_cols_5 = ['open', 'high', 'low', 'volume', 'close', 'ask1_price_trend_60', 'bid1_price_trend_60', 'bid1_size_n', 'bid2_size_n', 'bid3_size_n', 'bid4_size_n', 'bid5_size_n', 'ask1_size_n', 'ask2_size_n', 'ask3_size_n', 'ask4_size_n', 'ask5_size_n', 'log_return_bid1_price', 'log_return_bid2_price', 'log_return_ask1_price', 'log_return_ask2_price', 'buy_spread_trend_60', 'sell_spread_trend_60', 'wap_1_trend_60', 'wap_2_trend_60', 'buy_vwap_trend_60', 'sell_vwap_trend_60', 'volume_trend_60','minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']

    if i == 0:
        return feature_cols_0
    elif i == 1:
        return feature_cols_1
    elif i == 2:
        return feature_cols_2
    elif i == 3:
        return feature_cols_3
    elif i == 4:
        return feature_cols_4
    elif i == 5:
        return feature_cols_5
    else:
        raise ValueError("Invalid feature set index")