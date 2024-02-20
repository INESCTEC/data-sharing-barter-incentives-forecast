# flake8: noqa

import numpy as np
import pandas as pd


def process_dataset(dataset, tz):
    """
    Function to process dataset loaded from CSV.
    - Reindexes the dataset to contain timestamps with missing data
    - Removes holidays and bridges
    - Converts to desired timezone

    :param dataset: dataset to process
    :type dataset: pd.DataFrame
    :param tz: timezone to convert timestamps to
    :type tz: str
    :return: processed dataset
    :rtype: pd.DataFrame
    """
    expected_dates = pd.date_range(dataset.index[0],
                                   dataset.index[-1],
                                   freq='H',
                                   tz="UTC").tz_convert(tz)
    # Convert index to desired timezone
    _dataset = dataset.tz_localize('UTC').tz_convert(tz)
    _dataset = _dataset.reindex(expected_dates)
    return _dataset


def pearsoncorr(dataset, nlags):
    """
    Compute Pearson correlation with up to a desired number of lags.
    Shifts the dataset to obtain lags.
    Computes correlation between filtered dataset and each lag.

    :param dataset: target data to compute correlation
    :type dataset: pd.DataFrame
    :param dataset: target data to compute correlation
    :param nlags: limit of the range of lags to test
    :type nlags: None | int
    :return: Pearson correlation for each lag
    :rtype: list
    """
    corrs = [1]
    for i in range(1, nlags + 1):
        corrs.append(dataset["real"].corr(dataset["real"].shift(i)))
    return pd.DataFrame(corrs, columns=["acf"])


def compute_acf(dataset, forecast_horizon,
                nlags=None, select_top=None, threshold=None):
    """
    Function to compute and process the Pearson correlations of a dataset
    and its lags.
    Only outputs lags equal to or above 72 hours and allows to choose how to
    cut the level of correlations.

    :param dataset: target data to compute correlation
    :type dataset: pd.DataFrame
    :param nlags: limit of the range of lags to test
    :type nlags: None | int
    :param select_top: number of highest absolute correlation lags to select
    :type select_top: None | int
    :param threshold: minimum level of correlations to select lags
    :type threshold: None | int
    :return: dataframe with ordered lags and their correlations
    :rtype: pd.DataFrame
    """
    # Calculate Autocorrelation
    nlags = 744 if nlags is None else nlags
    coef = pearsoncorr(dataset=dataset, nlags=nlags)
    # Only correlations of lags equal or greater than horizon
    coef = coef.loc[coef.index >= forecast_horizon]
    # Cut autocorrelation (threshold or top values)
    if threshold:
        mask = coef.abs() > threshold
        thresh_lags = coef.loc[mask["acf"]]
        sorted_lags = thresh_lags.abs().sort_values(by='acf', ascending=False).index  # noqa
        thresh_sorted_lags = thresh_lags.reindex(sorted_lags)
        if select_top:
            result_lags = thresh_sorted_lags.iloc[:select_top]
        else:
            result_lags = thresh_sorted_lags
    elif select_top:
        sorted_lags = coef.abs().sort_values(by='acf', ascending=False).index
        result_lags = coef.reindex(sorted_lags).iloc[:select_top]
    else:
        result_lags = coef

    return result_lags


def filter_dataset(dataset, target_period):
    """
    Filter dataset according to periods of week in analysis.

    Possible target periods are:
    - "mon", "tue", "wed", "thu", "fri", "sat", "sun" - specific days of the week
    - "week, "weekend" - only weekdays or weekends
    - "mon-thu-fri", "tue-wed" - specific groups of the week

    :param dataset: target data to apply filter
    :type dataset: pd.DataFrame
    :param target_period: desired period of the week
    :type target_period: str
    :return: filtered dataset
    :rtype: pd.DataFrame
    """
    _dataset_filtered = dataset.copy()

    if target_period == "mon-thu-fri":
        _dataset_filtered.loc[
            ~np.isin(_dataset_filtered.index.weekday, [0, 3, 4])] = np.nan
    elif target_period == "tue-wed":
        _dataset_filtered.loc[
            ~np.isin(_dataset_filtered.index.weekday, [1, 2])] = np.nan
    elif target_period == "weekend":
        _dataset_filtered.loc[
            ~np.isin(_dataset_filtered.index.weekday, [5, 6])] = np.nan
    elif target_period == "week":
        _dataset_filtered.loc[
            np.isin(_dataset_filtered.index.weekday, [5, 6])] = np.nan
    elif target_period == "mon":
        _dataset_filtered.loc[
            np.isin(_dataset_filtered.index.weekday, [0])] = np.nan
    elif target_period == "tue":
        _dataset_filtered.loc[
            np.isin(_dataset_filtered.index.weekday, [1])] = np.nan
    elif target_period == "wed":
        _dataset_filtered.loc[
            np.isin(_dataset_filtered.index.weekday, [2])] = np.nan
    elif target_period == "thu":
        _dataset_filtered.loc[
            np.isin(_dataset_filtered.index.weekday, [3])] = np.nan
    elif target_period == "fri":
        _dataset_filtered.loc[
            np.isin(_dataset_filtered.index.weekday, [4])] = np.nan

    return _dataset_filtered


def compute_autocorrelations(dataset, forecast_horizon, acf_kwargs):
    """
    Function to run entire process of computing autocorrelation and
    partial autocorrelation functions.
    First, filters dataset according to desired period.
    Then, computes correlation with desired lags.
    Lastly, selected strongest lags are tested for partial autocorrelation.

    For now, only way to cut autocorrelation is by setting a threshold or
    defining a number of top lags to choose. These options can be combined.

    Possible target periods are:
    - "mon", "tue", "wed", "thu", "fri", "sat", "sun" - Compute correlations for specific days of the week
    - "week, "weekend" - only weekdays or weekends
    - "mon-thu-fri", "tue-wed" - specific groups of the week
    :param dataset:
    :type dataset:
    :param acf_kwargs: arguments for the acf method (see `compute_acf`)
    :type acf_kwargs: dict
    :param pacf_kwargs: arguments for the pacf method (see `compute_pacf`)
    :type pacf_kwargs: dict
    :param target_period: period of the week in analysis
    :type target_period: str
    :return: Output of the ACF and PACF functions
    :rtype: tuple
    """
    # Drop NaN values
    _dataset = dataset.dropna()

    # Compute ACF
    acf_coef = compute_acf(dataset=_dataset,
                           forecast_horizon=forecast_horizon,
                           **acf_kwargs)

    acf_coef = list(zip(acf_coef.index, acf_coef["acf"].values))
    return acf_coef


def autocorrelation_analysis(dataset,
                             acf_kwargs,
                             target_col,
                             forecast_horizon):
    _dataset = dataset[[target_col]].copy()
    _dataset.rename(columns={target_col: "real"}, inplace=True)
    target_autocorrelations = dict()
    acf_coef = compute_autocorrelations(
        dataset=_dataset,
        acf_kwargs=acf_kwargs,
        forecast_horizon=forecast_horizon
    )
    target_autocorrelations["general"] = {
        "acf": acf_coef
    }
    return target_autocorrelations
