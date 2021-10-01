import numpy as np
from copy import deepcopy


# #############################################################################
# IOTA unit conversions - REST API in IOTA / Market in MIOTA
# #############################################################################


def convert_mi_to_i(value_mi):
    """
    Convert from MIOTAs to IOTAs

    :param value_mi: Value in MIOTAs
    :return:
    """
    if not isinstance(value_mi, (int, float, np.float64)):
        raise TypeError("'value_mi' must be one of the following "
                        "types: [int, float, np.float64]")

    return np.float64(value_mi * 1000000.0)


def convert_i_to_mi(value_i):
    """
    Convert from IOTAs to MIOTAs

    :param value_i: Value in IOTAs
    :return:
    """
    if not isinstance(value_i, (int, float, np.float64)):
        raise TypeError("'value_mi' must be one of the following "
                        "types: [int, float, np.float64]")
    return np.float64(value_i / 1000000.0)


def convert_session_data_to_mi(data):
    data = deepcopy(data)
    data["b_min"] = convert_i_to_mi(data["b_min"])
    data["b_max"] = convert_i_to_mi(data["b_max"])
    data["market_price"] = convert_i_to_mi(data["market_price"])
    return data


def convert_buyers_bids_to_mi(bids):
    if not isinstance(bids, list):
        raise AttributeError("Error! 'bid_list' must have list type.")

    bids = deepcopy(bids)
    for i in range(len(bids)):
        bids[i]["bid_price"] = convert_i_to_mi(bids[i]["bid_price"])
        bids[i]["max_payment"] = convert_i_to_mi(bids[i]["max_payment"])
    return bids
