# flake8: noqa
import warnings
warnings.filterwarnings('ignore')

def process_buyers_avg(path_buyers):
    import pandas as pd
    df_buyers = pd.read_csv(path_buyers)[['session_id', 'resource_id', 'gain', 'has_to_pay']]
    df_avg_results = df_buyers.groupby('resource_id')[['gain', 'has_to_pay']].mean().reset_index()
    df_avg_results['norm_gain'] = abs(df_avg_results['gain'].values)/sum(abs(df_avg_results['gain'].values))
    df_avg_results['norm_to_pay'] = abs(df_avg_results['has_to_pay'].values)/sum(abs(df_avg_results['has_to_pay'].values)) 
    df_avg_results['difference'] = abs(df_avg_results['norm_gain'] - df_avg_results['norm_to_pay'])
    df_avg_results['resource_id'] = [str(i) for i in df_avg_results['resource_id'].values]
    return df_avg_results

def process_payment_session(path_buyers):
    import pandas as pd
    import copy
    df_buyers = pd.read_csv(path_buyers)[['session_id', 'resource_id', 'gain', 'has_to_pay']]
    df_buyers['resource_id'] = [str(i) for i in df_buyers['resource_id'].values]
    list_session_id = list(df_buyers.session_id.unique())
    for session_id in list_session_id:
        df_buyers_session = df_buyers[df_buyers.session_id == session_id]
        df_buyers_session['norm_gain'] = abs(df_buyers_session['gain'].values)/sum(abs(df_buyers_session['gain'].values))
        df_buyers_session['norm_to_pay'] = abs(df_buyers_session['has_to_pay'].values)/sum(abs(df_buyers_session['has_to_pay'].values)) 
        df_buyers_session['difference'] = abs(df_buyers_session['norm_gain'] - df_buyers_session['norm_to_pay'])
        df_buyers_session = df_buyers_session[["session_id", "resource_id", "difference"]]
        if session_id == 0:
            df_results = copy.deepcopy(df_buyers_session)
        else:
            df_results = pd.concat([df_results, copy.deepcopy(df_buyers_session)], axis=0)
    df_results['session_id'] = [str(i) for i in df_results['session_id'].values]
    return df_results
