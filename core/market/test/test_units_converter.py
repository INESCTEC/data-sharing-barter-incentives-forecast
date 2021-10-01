import pytest
import numpy as np

from market.core.helpers.units_helpers import (
    convert_session_data_to_mi,
    convert_buyers_bids_to_mi,
    convert_i_to_mi,
    convert_mi_to_i
)


def test_mi_to_i_functions():
    value_i = 1000000
    value_mi = value_i / 1000000
    assert value_mi == convert_i_to_mi(value_i=value_i)
    assert value_i == convert_mi_to_i(value_mi=value_mi)

    with pytest.raises(TypeError):
        convert_i_to_mi(value_i=None)
        convert_mi_to_i(value_mi=None)

    with pytest.raises(TypeError):
        convert_i_to_mi(value_i="bob")
        convert_mi_to_i(value_mi="bob")


def test_mi_to_i_session_data(session_data_json):
    # -- Check if original response types are as expected
    assert isinstance(session_data_json["b_min"], int)
    assert isinstance(session_data_json["b_max"], int)
    assert isinstance(session_data_json["market_price"], float)

    # -- Convert session data to mi:
    session_data = convert_session_data_to_mi(session_data_json)

    # -- Check if units are correct:
    for k, v in session_data_json.items():
        if k in ["b_min", "b_max", "market_price"]:
            assert v / 1000000.0 == session_data[k]
            assert isinstance(session_data[k], np.float64)
        else:
            assert v == session_data[k]
            assert isinstance(v, type(session_data[k]))


def test_mi_to_i_buyers_bids_data(buyers_bids_json):
    # -- Convert session data to mi:
    buyers_bids = convert_buyers_bids_to_mi(buyers_bids_json)

    for i in range(len(buyers_bids_json)):
        # -- Check if original response types are as expected
        assert isinstance(buyers_bids_json[i]["bid_price"], int)
        assert isinstance(buyers_bids_json[i]["max_payment"], int)

        # -- Check if units are correct:
        for k, v in buyers_bids_json[i].items():
            if k in ["bid_price", "max_payment"]:
                assert v / 1000000.0 == buyers_bids[i][k]
                assert isinstance(buyers_bids[i][k], np.float64)
            else:
                assert v == buyers_bids[i][k]
                assert isinstance(v, type(buyers_bids[i][k]))
