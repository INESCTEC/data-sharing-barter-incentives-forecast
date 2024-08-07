import numpy as np
from copy import deepcopy


# #############################################################################
# Crypto unit conversions - BASE / Transaction units
# #############################################################################


def convert_session_data_to_base_unit(data, convert_fn):
    data = deepcopy(data)
    data["b_min"] = convert_fn(data["b_min"])
    data["b_max"] = convert_fn(data["b_max"])
    data["market_price"] = convert_fn(data["market_price"])
    return data


def convert_buyers_bids_to_base_unit(bids, convert_fn):
    if not isinstance(bids, list):
        raise AttributeError("Error! 'bid_list' must have list type.")

    bids = deepcopy(bids)
    for i in range(len(bids)):
        bids[i]["bid_price"] = convert_fn(bids[i]["bid_price"])
        bids[i]["max_payment"] = convert_fn(bids[i]["max_payment"])
    return bids
