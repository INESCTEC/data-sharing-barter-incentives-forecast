# flake8: noqa
def process_forecasts_csv(path):
    import pandas as pd 
    df_report_forecast = pd.read_csv(path)
    df_report_forecast['datetime'] = pd.to_datetime(df_report_forecast['datetime']).dt.tz_localize(None)
    list_session_id = list(df_report_forecast.session_id.unique())
    list_df_prediction = []
    list_datetime_prediction = []
    for session_id in list_session_id:
        df_report_forecast_session = df_report_forecast[df_report_forecast.session_id == session_id]
        df_report_forecast_session_pivoted = df_report_forecast_session.pivot(index=["datetime"], columns="resource_id", values="value").reset_index()
        df_report_forecast_session_pivoted.columns.name = None
        df_prediction = df_report_forecast_session_pivoted.copy().set_index('datetime')
        df_prediction.columns = [str(col) for col in list(df_prediction.columns)]
        datetime_list = df_report_forecast_session_pivoted.datetime
        list_df_prediction.append(df_prediction)
        list_datetime_prediction.append(datetime_list)
    return list_session_id, list_df_prediction, list_datetime_prediction

def process_target_csv(path, datetime_list):
    import pandas as pd 
    df_measurements = pd.read_csv(path)
    df_measurements['datetime'] = pd.to_datetime(df_measurements['datetime'])
    df_target = df_measurements[df_measurements.datetime.isin(datetime_list)].set_index('datetime')
    return df_target

def metric_mape(y, y_pred):
    import numpy as np
    return np.mean(np.abs((y - y_pred)))

def performance_per_session(df_prediction, df_target, dict_performance):
    list_target_name = [col for col in list(df_target.columns)]
    for i, col in enumerate(list_target_name):
        pred = df_prediction[col].values
        targ = df_target[col].values
        mape = metric_mape(targ, pred)
        dict_performance[col] = mape
    return dict_performance

def forecast_performance(path_prediction, path_target):
    import pandas as pd
    from collections import defaultdict
    list_results = []
    dict_performance = defaultdict(dict)
    list_session_id, list_df_prediction, list_datetime_prediction = process_forecasts_csv(path_prediction)
    for session_id in list_session_id:
        datetime_list = list_datetime_prediction[session_id]
        df_prediction = list_df_prediction[session_id]
        df_target = process_target_csv(path_target, datetime_list)
        dict_performance = {'session_id' : session_id}
        dict_performance = performance_per_session(df_prediction, df_target, dict_performance)
        list_results.append(dict_performance)
    df_results = pd.DataFrame(list_results)
    return df_results



