# flake8: noqa
def melt_results(df_results):
    import pandas as pd
    df_melt = pd.melt(df_results, id_vars=['session_id'])
    df_melt.columns = ['session_id', 'resource_id', 'performance_error']
    df_melt['session_id'] = [str(i) for i in list(df_melt['session_id'].values)]
    return df_melt