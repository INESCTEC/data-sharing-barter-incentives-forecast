import pytest
import numpy as np
import datetime as dt

from src.market import (
    MarketClass,
    SessionClass,
    BuyerClass,
    SellerClass
)


@pytest.fixture
def expected_session_class_attrs():
    return sorted(['b_max', 'b_min', 'buyers_results', 'delta', 'epsilon',
                   'finish_ts', 'launch_ts', 'market_price', 'n_price_steps',
                   'next_market_price', 'next_weights_p', 'possible_p',
                   'prev_weights_p', 'sellers_results', 'session_date',
                   'session_id', 'session_number', 'status'])


@pytest.fixture
def expected_buyer_bid_fields():
    return sorted(['bid_price',
                   'gain_func',
                   'max_payment',
                   'user'])


@pytest.fixture
def market_class(market_session):
    mc = MarketClass(n_jobs=-1)
    mc.init_session(
        session_data=market_session.active_session,
        price_weights=market_session.price_weights,
        launch_time=market_session.launch_time
    )
    return mc


def test_market_class_init(market_session, market_class,
                           expected_session_class_attrs,):
    ms = market_session
    mc = market_class
    # Check if attr self.mkt_session is a SessionClass:
    assert isinstance(mc.mkt_sess, SessionClass)
    # Check if SessionClass has the expected attributes & attribute types:
    assert sorted(mc.mkt_sess.__dataclass_fields__) == expected_session_class_attrs  # noqa
    assert mc.mkt_sess.validate_attr_types()
    # Check if all the expected parameters were initialized:
    assert mc.mkt_sess.b_min == ms.active_session["b_min"]
    assert mc.mkt_sess.b_max == ms.active_session["b_max"]
    assert mc.mkt_sess.market_price == ms.active_session["market_price"]
    assert mc.mkt_sess.status == ms.active_session["status"]
    assert mc.mkt_sess.session_id == ms.active_session["market_session_id"]
    assert mc.mkt_sess.session_number == ms.active_session["session_number"]
    assert mc.mkt_sess.session_date == ms.active_session["session_date"]
    assert all(market_session.price_weights == market_session.price_weights)
    assert mc.mkt_sess.buyers_results == {}
    assert mc.mkt_sess.sellers_results == {}
    assert mc.mkt_sess.n_price_steps == ms.active_session["n_price_steps"]
    assert mc.mkt_sess.delta == ms.active_session["delta"]
    assert isinstance(mc.mkt_sess.possible_p, np.ndarray)
    assert isinstance(mc.mkt_sess.epsilon, np.float64)


def test_market_class_load_buyers_bids(market_class, bids_3b_3s,
                                       expected_buyer_bid_fields):
    mc = market_class
    buyers_bids, _ = bids_3b_3s
    # -- Load agents bids:
    mc.load_buyers_bids(bids=buyers_bids)
    # -- Check attributes values:
    assert isinstance(mc.buyers_data, dict)
    assert len(mc.buyers_data) == len(buyers_bids)
    for bid in buyers_bids:
        stored_bid = mc.buyers_data[bid["user"]]
        assert isinstance(stored_bid, BuyerClass)
        assert stored_bid.initial_bid == bid["bid_price"]
        assert stored_bid.gain_func == bid["gain_func"]
        assert stored_bid.max_payment == bid["max_payment"]
        assert stored_bid.identifier == bid["user"]
        assert stored_bid.has_to_pay == 0.0
        assert stored_bid.final_bid is None
        assert stored_bid.forecasts is None
        assert stored_bid.gain is None
        assert stored_bid.y is None


def test_market_class_load_sellers_bids(market_class, bids_3b_3s,
                                        expected_buyer_bid_fields):
    # todo: update this test after sellers bids updates
    mc = market_class
    _, sellers_list = bids_3b_3s
    # -- Load agents bids:
    mc.load_sellers(identifiers=sellers_list)
    # -- Check attributes values:
    assert isinstance(mc.sellers_data, dict)
    assert len(mc.sellers_data) == len(sellers_list)
    for seller_id in sellers_list:
        stored_bid = mc.sellers_data[seller_id]
        assert isinstance(stored_bid, SellerClass)
        assert stored_bid.identifier == seller_id
        assert stored_bid.has_to_receive == 0.0
        assert stored_bid.y is None


def test_load_agents_measurements(market_class, bids_3b_3s,
                                  measurements_bids_3b_3s):
    mc = market_class
    buyers_bids, sellers_list = bids_3b_3s
    # -- Load agents bids:
    mc.load_buyers_bids(bids=buyers_bids)
    mc.load_sellers(identifiers=sellers_list)
    # -- Load agents measurements data:
    mc.load_agents_measurements(measurements=measurements_bids_3b_3s)

    for agent_id, measur_df in measurements_bids_3b_3s.items():
        if agent_id in mc.buyers_data:
            assert all(mc.buyers_data[agent_id].y == measur_df)
        if agent_id in mc.sellers_data:
            assert all(mc.sellers_data[agent_id].y == measur_df)


def test_run_market_session_no_fees(market_class,
                                    bids_3b_3s,
                                    measurements_bids_3b_3s):
    mc = market_class
    mc.MARKET_FEE_PCT = 0.0
    buyers_bids, sellers_list = bids_3b_3s
    # -- Load agents bids:
    mc.load_buyers_bids(bids=buyers_bids)
    mc.load_sellers(identifiers=sellers_list)
    # -- Load agents measurements data:
    mc.load_agents_measurements(measurements=measurements_bids_3b_3s)
    # -- Start market session:
    # todo: integration with api_controller
    mc.start_session(api_controller=None)
    assert mc.mkt_sess.status == "running"
    mc.run_session()
    # Check if the necessary data was created:
    assert isinstance(mc.mkt_sess.buyers_results, dict)
    assert isinstance(mc.mkt_sess.sellers_results, dict)
    assert len(mc.mkt_sess.buyers_results) == len(buyers_bids)
    assert len(mc.mkt_sess.sellers_results) == len(sellers_list)
    for buyer, res in mc.mkt_sess.buyers_results.items():
        assert sorted(res.keys()) == ['final_bid',
                                      'gain',
                                      'gain_func',
                                      'has_to_pay',
                                      'identifier',
                                      'initial_bid',
                                      'max_payment']
        assert len([k for k, v in res.items() if v is None]) == 0
        assert res["has_to_pay"] >= 0

    for seller, res in mc.mkt_sess.sellers_results.items():
        assert sorted(res.keys()) == ['has_to_receive', 'identifier']
        assert len([k for k, v in res.items() if v is None]) == 0
        assert res["has_to_receive"] >= 0

    # total agent payments:
    payments = [v["has_to_pay"] for x, v in mc.mkt_sess.buyers_results.items()]
    total_payments = sum(payments)
    # total agent revenues:
    revenues = [v["has_to_receive"] for x, v in mc.mkt_sess.sellers_results.items()]  # noqa
    total_revenues = sum(revenues)
    # Since there are no market fees -> all payment from buyers goes to sellers
    # therefore, sum(payments) == sum(revenues)
    assert round(total_payments, 6) == round(total_revenues, 6)
    assert mc.mkt_sess.total_market_fee == 0.0
    assert mc.mkt_sess.market_fee_per_buyer == {0: 0.0, 1: 0.0, 2: 0.0}
    assert sum(mc.mkt_sess.market_fee_per_buyer.values()) == mc.mkt_sess.total_market_fee  # noqa


def test_run_market_session_with_fees(market_class,
                                      bids_3b_3s,
                                      measurements_bids_3b_3s):
    mc = market_class
    mc.MARKET_FEE_PCT = 0.15
    buyers_bids, sellers_list = bids_3b_3s
    # -- Load agents bids:
    mc.load_buyers_bids(bids=buyers_bids)
    mc.load_sellers(identifiers=sellers_list)
    # -- Load agents measurements data:
    mc.load_agents_measurements(measurements=measurements_bids_3b_3s)
    # -- Start market session:
    # todo: integration with api_controller
    mc.start_session(api_controller=None)
    assert mc.mkt_sess.status == "running"
    mc.run_session()
    # Check if the necessary data was created:
    assert isinstance(mc.mkt_sess.buyers_results, dict)
    assert isinstance(mc.mkt_sess.sellers_results, dict)
    assert len(mc.mkt_sess.buyers_results) == len(buyers_bids)
    assert len(mc.mkt_sess.sellers_results) == len(sellers_list)
    for buyer, res in mc.mkt_sess.buyers_results.items():
        assert sorted(res.keys()) == ['final_bid',
                                      'gain',
                                      'gain_func',
                                      'has_to_pay',
                                      'identifier',
                                      'initial_bid',
                                      'max_payment']
        assert len([k for k, v in res.items() if v is None]) == 0
        assert res["has_to_pay"] >= 0

    for seller, res in mc.mkt_sess.sellers_results.items():
        assert sorted(res.keys()) == ['has_to_receive', 'identifier']
        assert len([k for k, v in res.items() if v is None]) == 0
        assert res["has_to_receive"] >= 0

    # total agent payments:
    payments = [v["has_to_pay"] for x, v in mc.mkt_sess.buyers_results.items()]
    total_payments = sum(payments)
    # total agent revenues:
    revenues = [v["has_to_receive"] for x, v in mc.mkt_sess.sellers_results.items()]  # noqa
    total_revenues = sum(revenues)
    # Since there are market fees -> buyers payment goes to sellers & market
    # therefore, sum(payments) == sum(revenues) + market_fee
    assert round(total_payments, 6) == round(total_revenues + mc.mkt_sess.total_market_fee, 6) # noqa
    # print("Total payments:", total_payments)
    # print("Total revenues:", total_revenues)
    # print("Total market fee:", mc.mkt_sess.total_market_fee)
    # print("Rounded payments:", round(total_payments, 5))
    # print("Rounded revenue + market fee:",
    #       round(mc.mkt_sess.total_market_fee, 5))
    assert mc.mkt_sess.total_market_fee > 0.0
    assert sum(mc.mkt_sess.market_fee_per_buyer.values()) == mc.mkt_sess.total_market_fee  # noqa
