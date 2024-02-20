import pytest
import numpy as np

from src.market.helpers.units_helpers import (
    convert_session_data_to_mi,
    convert_buyers_bids_to_mi,
    convert_i_to_mi,
    convert_mi_to_i
)

from ..common import (
    create_market_session_data,
    create_user_resource_db,
    create_buyer_bid_per_resource
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


def test_mi_to_i_session_data():
    # Session data in original units:
    session_data = create_market_session_data(convert_to_miota=False)

    # -- Check if original response types are as expected
    assert isinstance(session_data["b_min"], int)
    assert isinstance(session_data["b_max"], int)
    assert isinstance(session_data["market_price"], float)

    # -- Convert session data to mi:
    session_data_mi = convert_session_data_to_mi(session_data)

    # -- Check if units are correct:
    for k, v in session_data.items():
        if k in ["b_min", "b_max", "market_price"]:
            assert v / 1000000.0 == session_data_mi[k]
            assert isinstance(session_data_mi[k], np.float64)
        else:
            assert v == session_data_mi[k]
            assert isinstance(v, type(session_data_mi[k]))


def test_mi_to_i_buyers_bids_data():
    resource_db = create_user_resource_db(nr_users=3, nr_resources_per_user=3)
    bid_db = create_buyer_bid_per_resource(resource_db=resource_db,
                                           convert_to_miota=False)

    # -- Convert session data to mi:
    buyers_bids = convert_buyers_bids_to_mi(bid_db)

    for i in range(len(bid_db)):
        # -- Check if original response types are as expected
        assert isinstance(bid_db[i]["bid_price"], int)
        assert isinstance(bid_db[i]["max_payment"], int)

        # -- Check if units are correct:
        for k, v in bid_db[i].items():
            if k in ["bid_price", "max_payment"]:
                assert v / 1000000.0 == buyers_bids[i][k]
                assert isinstance(buyers_bids[i][k], np.float64)
            else:
                assert v == buyers_bids[i][k]
                assert isinstance(v, type(buyers_bids[i][k]))
