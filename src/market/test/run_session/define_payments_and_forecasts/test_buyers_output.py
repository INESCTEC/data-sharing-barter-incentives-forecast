
def test_buyers_output_attrs(market_session_2u_1r):
    mc = market_session_2u_1r
    # Run market session:
    mc.define_payments_and_forecasts()
    # Check if buyer outputs were properly created:
    assert len(mc.buyer_outputs) > 0
    buyer_output = mc.buyer_outputs[0]
    buyer_output_fields = buyer_output.keys()
    import numpy as np
    assert isinstance(buyer_output["features"], np.ndarray)
    assert buyer_output["features"].shape == buyer_output["noisy_train_features"].shape
    assert sorted(buyer_output_fields) == sorted(['features',
                                                  'noisy_train_features',
                                                  'market_fee',
                                                  'payment',
                                                  'targets',
                                                  'gain_func',
                                                  'gain',
                                                  'final_bid',
                                                  'initial_bid',
                                                  'resource_id',
                                                  'user_id',
                                                  'sellers_features_name'])


