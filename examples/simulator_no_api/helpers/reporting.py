import os
import pandas as pd


def create_report(session_id,
                  session_lt,
                  buyers_df,
                  sellers_df,
                  market_df,
                  market_sess,
                  path):
    sess_dict = {}
    for k, v in market_sess.details.items():
        sess_dict[k] = str(v)
    sess_dict["session_id"] = session_id
    sess_dict["session_lt"] = session_lt
    market_df = market_df.append(pd.DataFrame(sess_dict, index=[0]),
                                 ignore_index=True)
    for bb in market_sess.buyers_results.values():
        bb["session_id"] = session_id
        bb["session_lt"] = session_lt
        buyers_df = buyers_df.append(pd.DataFrame(bb, index=[0]),
                                     ignore_index=True)

    for ss in market_sess.sellers_results.values():
        ss["session_id"] = session_id
        ss["session_lt"] = session_lt
        sellers_df = sellers_df.append(pd.DataFrame(ss, index=[0]),
                                       ignore_index=True)

    # Update report files:
    market_df.to_csv(os.path.join(path, "market.csv"), index=False)
    buyers_df.to_csv(os.path.join(path, "buyers.csv"), index=False)
    sellers_df.to_csv(os.path.join(path, "sellers.csv"), index=False)
    return buyers_df, sellers_df, market_df
