# flake8: noqa
def read_market_elapsed_time(path_std, path_fs):
    import pandas as pd
    df_time_elapsed = pd.read_csv(path_std)[['session_id', 'elapsed_time']].set_index("session_id")
    df_time_elapsed.columns = ['elaps_time_std']
    df_time_elapsed_fs = pd.read_csv(path_fs)[['session_id', 'elapsed_time']].set_index("session_id")
    df_time_elapsed_fs.columns = ['elaps_time_fs']
    df_performance = pd.concat([df_time_elapsed, df_time_elapsed_fs], axis=1).reset_index()
    return df_performance