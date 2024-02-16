# flake8: noqa
def process_coefs_csv(path_coefs):
    import pandas as pd
    df_coefs = pd.read_csv(path_coefs, index_col=0)
    list_feat = list(df_coefs.index)
    return df_coefs, list_feat

def process_sellers_csv(path_sellers, list_feat):
    import pandas as pd
    df = pd.read_csv(path_sellers)
    df_sellers = df[df.resource_id.isin(list_feat)]
    return df_sellers[["session_id", "resource_id", "has_to_receive"]]

def process_sellers_session(df_sellers):
    import copy
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    list_session_id = list(df_sellers.session_id.unique())
    for session_id in list_session_id:
        df_sellers_session = df_sellers[df_sellers.session_id == session_id]
        df_sellers_session['norm_to_receive'] = abs(df_sellers_session['has_to_receive'].values)/sum(abs(df_sellers_session['has_to_receive'].values)) 
        if session_id == 0:
            df_revenue_session = copy.deepcopy(df_sellers_session)
        else:
            df_revenue_session = pd.concat([df_revenue_session, copy.deepcopy(df_sellers_session)], axis=0)
    return df_revenue_session

def process_sellers_avg_csv(path_sellers, list_feat):
    import pandas as pd
    df = pd.read_csv(path_sellers)
    df_sellers = df[df.resource_id.isin(list_feat)]
    df_sellers_revenue_avg = df_sellers.groupby('resource_id')['has_to_receive'].mean()
    return df_sellers_revenue_avg

def normalize_coefs(df_coefs):
    df_coefs_sum = df_coefs.sum(axis=1)
    df_coefs_norm = df_coefs_sum/sum(df_coefs_sum)
    df_coefs_norm.name = 'norm_coefs'
    return df_coefs_norm

def normalize_sellers(df_sellers_revenue_avg):
    df_sellers_norm = round(df_sellers_revenue_avg/sum(df_sellers_revenue_avg), 6)
    df_sellers_norm.name = 'norm_to_receive' 
    return df_sellers_norm

def merge_dfs(df_sellers_norm, df_coefs_norm):
    import pandas as pd
    df_results = pd.concat([df_sellers_norm, df_coefs_norm], axis=1).reset_index()
    df_results.columns = ['resource_id', 'norm_to_receive' , 'norm_coefs']
    df_results['difference'] = abs(df_results['norm_to_receive'] - df_results['norm_coefs'])
    return df_results

def revenue_avg(path_coefs, path_sellers):
    df_coefs, list_feat = process_coefs_csv(path_coefs)
    df_sellers_revenue_avg = process_sellers_avg_csv(path_sellers, list_feat)
    df_coefs_norm = normalize_coefs(df_coefs)
    df_sellers_norm = normalize_sellers(df_sellers_revenue_avg)
    df_results = merge_dfs(df_sellers_norm, df_coefs_norm)
    df_results['resource_id'] = [str(i) for i in list(df_results['resource_id'].values)]
    return df_results

def merge_session(df_seller_session, df_coefs_norm):
    df_coefs_norm = df_coefs_norm.reset_index()
    df_coefs_norm.columns = ["resource_id", "norm_coefs"]
    df_session = df_seller_session.set_index("resource_id").join(df_coefs_norm.set_index("resource_id"), on="resource_id").reset_index()
    df_session['difference'] = abs(df_session['norm_coefs'] - df_session['norm_to_receive'])
    df_session['resource_id'] = [str(i) for i in list(df_session['resource_id'].values)]
    df_session['session_id'] = [str(i) for i in list(df_session['session_id'].values)]
    return df_session

def revenue_session(path_coefs, path_sellers):
    df_coefs, list_feat = process_coefs_csv(path_coefs)
    df_coefs_norm = normalize_coefs(df_coefs)
    df_sellers = process_sellers_csv(path_sellers, list_feat)
    df_seller_session = process_sellers_session(df_sellers)
    df_session = merge_session(df_seller_session, df_coefs_norm)
    return df_session


