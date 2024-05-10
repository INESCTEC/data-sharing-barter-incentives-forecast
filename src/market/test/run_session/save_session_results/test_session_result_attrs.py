
def test_buyers_and_sellers_results_attrs(market_session_2u_2r):
    mc = market_session_2u_2r
    # Run market session:
    mc.define_payments_and_forecasts()
    mc.define_sellers_revenue()
    mc.save_session_results()

    # Check if the necessary data was created:
    assert isinstance(mc.mkt_sess.buyers_results, dict)
    assert isinstance(mc.mkt_sess.sellers_results, dict)
    assert len(mc.mkt_sess.buyers_results) == 4
    assert len(mc.mkt_sess.sellers_results) == 4

    # Check if necessary variables are available:
    for buyer, res in mc.mkt_sess.buyers_results.items():
        assert sorted(res.keys()) == ['features_list',
                                      'final_bid',
                                      'gain',
                                      'gain_func',
                                      'has_to_pay',
                                      'initial_bid',
                                      'max_payment',
                                      'resource_id',
                                      'user_id']
        assert len([k for k, v in res.items() if v is None]) == 0
        assert res["has_to_pay"] >= 0

    for seller, res in mc.mkt_sess.sellers_results.items():
        assert sorted(res.keys()) == ['has_to_receive',
                                      'resource_id',
                                      'shapley_value',
                                      'user_id']
        assert len([k for k, v in res.items() if v is None]) == 0
        assert res["has_to_receive"] >= 0
