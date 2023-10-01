import numpy as np
import pandas as pd

from math import sqrt

from sklearn.metrics import (
    mean_absolute_error as _mae_,
    mean_squared_error as _mse_
)

from src.market.models.linear import linear_regression as forecast_model


""" -------------- BASE EVALUATION METRICS -------------- """


def __mae(real, pred):
    return _mae_(real, pred)


def __rmse(real, pred):
    return sqrt(_mse_(real, pred))


def __mse(real, pred):
    return _mse_(real, pred)


def calc_forecast_error(X, y, n_hours, gain_func="rmse"):
    # todo: @Ricardo 1) mudar forma como é selecionado conjunto de treino/teste
    #  Garantir que funciona para conjuntos com menos de N_HOURS disponiveis
    # Train/test split
    X_train = X[0:(X.shape[0] - n_hours - 1), :]
    X_val = X[(X.shape[0] - n_hours):, :]
    y_train = y[0:(X.shape[0] - n_hours - 1), :]
    y_val = y[(X.shape[0] - n_hours):, :]

    # Generate forecasts:
    preds = forecast_model(X_train, X_val, y_train)

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
    np.random.seed(1)  # todo: @Ricardo remove this seed
    sigma_feat = 0.5 * features.std(axis=0)[:-1]  # todo: mult. por 0.5?
    noise = np.zeros(shape=features.shape)
    for i, sigma in enumerate(sigma_feat):
        noise[:, i] = np.random.normal(0, sigma, (features.shape[0], ))
    return noise


def generate_noise(features):
    np.random.seed(1)  # todo: @Ricardo remove this seed
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
    # 1) Train & evaluate model w/ buyer features
    buyer_err = calc_forecast_error(
        X=features[:, buyer_feature_pos].reshape(-1, 1),
        y=targets,
        n_hours=n_hours,
        gain_func=gain_func,
    )
    # 2) Train & evaluate model w/ buyer + market features
    # input_x = market_x.join(buyer_x)
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
        features,
        targets,
        gain_func,
        n_hours: int,
        market_price: float,
        b_max: float,
        bid_price: float,
):

    # noise = generate_noise(features=features)
    noise = generate_noise_per_feature(features=features)
    # todo: rever ratio
    # Nota1: Versão carla acaba por adicionar mt ruido para diferenças % baixas
    # em valores elevados de market price / bid price
    # -- Versão Carla
    ratio_ = max(0, market_price - bid_price)
    # -- Nova Versão:
    # Nota2: Versão nova parece penalizar pouco estas diferenças, especialmente
    # no cálculo de ganho para varios niveis de preço
    # Ver variavel I_ -> metodo calc_buyer_payment()
    # ratio_ = max(0, market_price - bid_price) / b_max
    # ratio_ = max(0, 1 - bid_price / market_price) * b_max
    noisy_features = (features + ratio_ * noise)
    gain = calculate_gain(
        features=noisy_features,
        targets=targets,
        gain_func=gain_func,
        n_hours=n_hours
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
    X_train = train_features

    # Generate forecasts:
    preds = forecast_model(X_train, X_forecast, train_targets)

    # Compute forecasts:
    forecasts_df["value"] = preds.ravel()
    forecasts_df.index.name = "datetime"
    return forecasts_df
