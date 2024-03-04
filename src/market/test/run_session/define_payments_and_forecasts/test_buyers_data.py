
def test_buyers_data_2u_1r(market_session_2u_1r):
    mc = market_session_2u_1r
    # Run market session:
    mc.define_payments_and_forecasts()
    # Check buyers_data attr:
    assert len(mc.buyers_data.keys()) == 2   # 2 resources (1 per user)
    assert list(mc.buyers_data.keys()) == ["0", "1"]
    assert list(mc.buyers_data.keys()) == ["0", "1"]

    for i, data in mc.buyers_data.items():
        assert data.resource_id == i
        assert data.y.shape[0] > 0
        assert list(data.y.columns) == ["value", "variable"]
        assert data.initial_bid >= data.final_bid


def test_buyers_data_2u_2r(market_session_2u_2r):
    mc = market_session_2u_2r
    # Run market session:
    mc.define_payments_and_forecasts()
    # Check buyers_data attr:
    assert len(mc.buyers_data.keys()) == 4   # 4 resources (2 per user)
    assert list(mc.buyers_data.keys()) == ["0", "1", "2", "3"]

    for i, data in mc.buyers_data.items():
        assert data.resource_id == i
        assert data.y.shape[0] > 0
        assert list(data.y.columns) == ["value", "variable"]
        assert data.initial_bid >= data.final_bid
