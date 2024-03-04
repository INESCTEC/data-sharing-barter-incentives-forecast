import numpy as np
import pandas as pd

from math import sqrt

from sklearn.metrics import (
    mean_absolute_error as _mae_,
    mean_squared_error as _mse_
)

from src.market.models.linear import linear_regression_pipe as forecast_model
from src.market.models.linear import cv_forecast, recent_hours_forecast


""" -------------- BASE EVALUATION METRICS -------------- """


def __mae(real, pred):
    return _mae_(real, pred)


def __rmse(real, pred):
    return sqrt(_mse_(real, pred))


def __mse(real, pred):
    return _mse_(real, pred)


def calc_forecast_error(X, y, n_hours, gain_func="rmse"):
    # Init forecast pipeline:
    pipe = forecast_model()

    # Compute forecasts (for recent hours or CV):
    # preds, y_val = cv_forecast(X, y, f_pipeline=pipe)
    preds, y_val = recent_hours_forecast(X, y, n_hours, f_pipeline=pipe)

    # Calculate gain (forecast error)
    if gain_func == "rmse":
        return __rmse(y_val, preds)
    elif gain_func == "mae":
        return __mae(y_val, preds)
    elif gain_func == "mse":
        return __mse(y_val, preds)


def calc_gain(buyer_err, market_err, targets):
    g = (buyer_err - market_err) / (np.max(targets) - np.min(targets))
    return max(0, g.mean()) * 100


def generate_noise_per_feature(features):
    # np.random.seed(1)  # Disable for production
    # sigma_feat = 0.5 * features.std(axis=0)  # Versao carla
    sigma_feat = features.std(axis=0)
    noise = np.zeros(shape=features.shape)
    for i, sigma in enumerate(sigma_feat):
        noise[:, i] = np.random.normal(0, sigma, (features.shape[0], ))
    return noise


def generate_noise(features):
    # np.random.seed(1)  # Disable for production
    sigma = 0.5 * pd.DataFrame(features).std(axis=0).mean()
    noise = np.random.normal(0, sigma, features.shape)
    noise[:, -1] = 0
    return noise


def calculate_gain(
        features,
        targets,
        gain_func,
        n_hours: int,
        buyer_feature_pos=-1,
        market_features_pos=None,
):
    """
    Calculate market gain to this buyer.

    :param features: Buyer features matrix
    :param targets: Buyer targets array
    :param gain_func: Buyer gain function
    :param n_hours: Number of hours for lookback evaluation period
    :param buyer_feature_pos: Buyer features position in matrix
    :param market_features_pos: Market feature position in matrix. If declared,
    assures that model is exclusively trained with that feature. Else, market
    model is trained with all market features (+ buyer feature).

    :return:
    """
    features = features.copy()

    # 1) Train & evaluate model w/ buyer features
    if len(buyer_feature_pos) > 1:
        buyer_feat = features[:, buyer_feature_pos]
    else:
        buyer_feat = features[:, buyer_feature_pos].reshape(-1, 1)

    buyer_err = calc_forecast_error(
        X=buyer_feat,
        y=targets,
        n_hours=n_hours,
        gain_func=gain_func,
    )
    # 2) Train & evaluate model w/ buyer + market features
    if any([x in buyer_feature_pos for x in market_features_pos]):
        raise ValueError("Market features cannot be the "
                         "same as buyer features.")

    if market_features_pos is not None:
        pos_ = np.append(market_features_pos, buyer_feature_pos)
        features = features[:, pos_]

    market_err = calc_forecast_error(
        X=features,
        y=targets,
        n_hours=n_hours,
        gain_func=gain_func,
    )
    # Calculate gain:
    gain = calc_gain(
        buyer_err=buyer_err,
        market_err=market_err,
        targets=targets
    )
    return gain


def calculate_noise_and_gain(
        bid_price: float,
        features,
        targets,
        gain_func,
        n_hours: int,
        market_price: float,
        b_max: float,
        buyer_features_idx,
        sellers_features_idx,
):

    # noise = generate_noise(features=features)
    sellers_features = features[:, sellers_features_idx]
    noise = generate_noise_per_feature(features=sellers_features)

    # todo: rever ratio
    # Nota1: Versão carla acaba por adicionar mt ruido para diferenças % baixas
    # em valores elevados de market price / bid price
    # -- Versão Carla
    # ratio_ = max(0, market_price - bid_price)
    # -- Nova Versão:
    # Nota2: Versão nova parece penalizar pouco estas diferenças, especialmente
    # no cálculo de ganho para varios niveis de preço
    # Ver variavel I_ -> metodo calc_buyer_payment()
    ratio_ = max(0, market_price - bid_price) / market_price
    # ratio_ = max(0, 1 - bid_price / market_price) * b_max
    noisy_sellers_features = (sellers_features + ratio_ * noise)

    noisy_features = features.copy()
    noisy_features[:, sellers_features_idx] = noisy_sellers_features

    gain = calculate_gain(
        features=noisy_features,
        targets=targets,
        gain_func=gain_func,
        n_hours=n_hours,
        buyer_feature_pos=buyer_features_idx,
        market_features_pos=sellers_features_idx,
    )
    return noisy_features, gain


def create_forecast_mock(features, targets, start_date, end_date):
    _st = start_date.replace(second=0, microsecond=0)
    _et = end_date.replace(second=0, microsecond=0)
    _ts = pd.date_range(_st, _et, freq="H", tz="utc")
    _v = np.random.uniform(low=0, high=1, size=len(_ts))
    forecasts = pd.DataFrame({
        "datetime": _ts,
        "value": _v,
        "units": ["w"] * len(_ts),
    })
    return forecasts


def create_forecast(train_features, train_targets, test_features_df):
    forecasts_df = pd.DataFrame(index=test_features_df.index)
    X_forecast = test_features_df.values

    # Generate forecasts:
    pipe = forecast_model()
    pipe.fit(train_features, train_targets)
    forecasts = pipe.predict(X_forecast)

    # Compute forecasts:
    forecasts_df["value"] = forecasts.ravel()
    forecasts_df.index.name = "datetime"
    return forecasts_df
