

def test_payment_vs_revenue_no_fees_2u_2r(market_session_2u_2r):
    mc = market_session_2u_2r
    mc.MARKET_FEE_PCT = 0.0
    # Run market session:
    mc.define_payments_and_forecasts()
    mc.define_sellers_revenue()
    mc.save_session_results()

    # total agent payments:
    payments = [v["has_to_pay"] for x, v in mc.mkt_sess.buyers_results.items()]
    total_payments = sum(payments)
    # total agent revenues:
    revenues = [v["has_to_receive"] for x, v in mc.mkt_sess.sellers_results.items()]
    total_revenues = sum(revenues)
    # Since there are no market fees -> all payment from buyers goes to sellers
    # therefore, sum(payments) == sum(revenues)
    assert round(total_payments, 6) == round(total_revenues, 6)
    assert mc.mkt_sess.total_market_fee == 0.0
    assert mc.mkt_sess.market_fee_per_resource == {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    assert sum(mc.mkt_sess.market_fee_per_resource.values()) == mc.mkt_sess.total_market_fee  # noqa


def test_payment_vs_revenue_no_fees_3u_4r(market_session_3u_4r):
    mc = market_session_3u_4r
    mc.MARKET_FEE_PCT = 0.0
    # Run market session:
    mc.define_payments_and_forecasts()
    mc.define_sellers_revenue()
    mc.save_session_results()

    # total agent payments:
    payments = [v["has_to_pay"] for x, v in mc.mkt_sess.buyers_results.items()]
    total_payments = sum(payments)
    # total agent revenues:
    revenues = [v["has_to_receive"] for x, v in mc.mkt_sess.sellers_results.items()]
    total_revenues = sum(revenues)
    # Since there are no market fees -> all payment from buyers goes to sellers
    # therefore, sum(payments) == sum(revenues)
    assert round(total_payments, 6) == round(total_revenues, 6)
    assert mc.mkt_sess.total_market_fee == 0.0
    assert mc.mkt_sess.market_fee_per_resource == {0: 0.0, 1: 0.0, 2: 0.0,
                                                   3: 0.0, 4: 0.0, 5: 0.0,
                                                   6: 0.0, 7: 0.0, 8: 0.0,
                                                   9: 0.0, 10: 0.0, 11: 0.0}
    assert sum(mc.mkt_sess.market_fee_per_resource.values()) == mc.mkt_sess.total_market_fee


def test_payment_vs_revenue_low_fees_2u_2r(market_session_2u_2r):
    mc = market_session_2u_2r
    mc.MARKET_FEE_PCT = 0.05  # 5% per payment of each resource
    # Run market session:
    mc.define_payments_and_forecasts()
    mc.define_sellers_revenue()
    mc.save_session_results()

    # total agent payments:
    payments = [v["has_to_pay"] for x, v in mc.mkt_sess.buyers_results.items()]
    total_payments = sum(payments)
    # total agent revenues:
    revenues = [v["has_to_receive"] for x, v in mc.mkt_sess.sellers_results.items()]
    total_revenues = sum(revenues)
    # Since there are market fees -> buyers payment goes to sellers & market
    # therefore, sum(payments) == sum(revenues) + market_fee
    assert round(total_payments, 6) == round(total_revenues + mc.mkt_sess.total_market_fee, 6)
    assert mc.mkt_sess.total_market_fee > 0.0
    assert sum(mc.mkt_sess.market_fee_per_resource.values()) == mc.mkt_sess.total_market_fee


def test_payment_vs_revenue_low_fees_3u_4r(market_session_3u_4r):
    mc = market_session_3u_4r
    mc.MARKET_FEE_PCT = 0.05  # 5% per payment of each resource
    # Run market session:
    mc.define_payments_and_forecasts()
    mc.define_sellers_revenue()
    mc.save_session_results()

    # total agent payments:
    payments = [v["has_to_pay"] for x, v in mc.mkt_sess.buyers_results.items()]
    total_payments = sum(payments)
    # total agent revenues:
    revenues = [v["has_to_receive"] for x, v in mc.mkt_sess.sellers_results.items()]
    total_revenues = sum(revenues)
    # Since there are market fees -> buyers payment goes to sellers & market
    # therefore, sum(payments) == sum(revenues) + market_fee
    assert round(total_payments, 6) == round(total_revenues + mc.mkt_sess.total_market_fee, 6)
    assert mc.mkt_sess.total_market_fee > 0.0
    assert sum(mc.mkt_sess.market_fee_per_resource.values()) == mc.mkt_sess.total_market_fee