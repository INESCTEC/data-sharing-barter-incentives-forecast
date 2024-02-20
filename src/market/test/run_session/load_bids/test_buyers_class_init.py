
from src.market import BuyerClass

from ...common import (
    create_buyer_bid_per_resource,
    create_user_resource_db
)


def expected_buyer_bid_fields():
    return sorted([
        'resource',
        'bid_price',
        'gain_func',
        'max_payment',
        'user',
        'market_bid_id',
    ])


def test_buyer_class_load_bids(session_class):
    resource_db = create_user_resource_db(nr_users=3, nr_resources_per_user=3)
    bid_db = create_buyer_bid_per_resource(resource_db=resource_db)

    session_class.load_users_resources(users_resources=resource_db)
    session_class.load_resources_bids(bids=bid_db)

    # -- Check attributes values:
    assert isinstance(session_class.buyers_data, dict)
    assert len(session_class.buyers_data) == len(bid_db)
    for bid in bid_db:
        resource_id = bid["resource"]
        stored_bid = session_class.buyers_data[resource_id]
        assert isinstance(stored_bid, BuyerClass)
        assert stored_bid.market_bid_id == bid["id"]
        assert stored_bid.user_id == bid["user"]
        assert stored_bid.resource_id == bid["resource"]
        assert stored_bid.initial_bid == bid["bid_price"]
        assert stored_bid.gain_func == bid["gain_func"]
        assert stored_bid.max_payment == bid["max_payment"]
        assert stored_bid.has_to_pay == 0.0
        assert stored_bid.final_bid is None
        assert stored_bid.forecasts is None
        assert stored_bid.gain is None
        assert stored_bid.y is None
        assert stored_bid.payment_split == {}

