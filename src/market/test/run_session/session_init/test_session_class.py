import numpy as np

from src.market import SessionClass


def expected_session_attrs():
    return sorted([
        # params:
        'b_max', 'b_min',
        'delta', 'epsilon',
        'possible_p',
        'next_weights_p', 'prev_weights_p', 'n_price_steps',
        # prices:
        'market_price',
        'next_market_price',
        # Results:
        'buyers_results', 'sellers_results',
        # Forecasts:
        'buyers_forecasts',
        # Timestamps / status
        'finish_ts', 'launch_ts',
        'status',
        # meta:
        'session_date', 'session_id', 'session_number',
    ])


###############################################################################
#  Tests Cases:
###############################################################################

def test_session_class_type(session_class):
    # Check if attr self.mkt_session is a SessionClass:
    assert isinstance(session_class.mkt_sess, SessionClass)


def test_session_class_attrs_name(session_class):
    # Check if SessionClass has the expected attributes & attribute types:
    current_fields = session_class.mkt_sess.__dataclass_fields__
    assert sorted(current_fields) == expected_session_attrs()
    assert session_class.mkt_sess.validate_attr_types()


def test_session_class_base_attrs(session_class, session_data):
    mc_sess = session_class.mkt_sess
    # Check if all the expected parameters were initialized:
    assert mc_sess.b_min == session_data["b_min"]
    assert mc_sess.b_max == session_data["b_max"]
    assert mc_sess.market_price == session_data["market_price"]
    assert mc_sess.status == session_data["status"]
    assert mc_sess.session_id == session_data["id"]
    assert mc_sess.session_number == session_data["session_number"]
    assert mc_sess.session_date.strftime("%Y-%m-%d") == session_data["session_date"]
    assert mc_sess.buyers_results == {}
    assert mc_sess.sellers_results == {}
    assert mc_sess.n_price_steps == session_data["n_price_steps"]
    assert mc_sess.delta == session_data["delta"]


def test_session_class_price_weight_attrs(session_class, price_weights):
    mc_sess = session_class.mkt_sess
    # Check  if previous weights list was initialized correctly:
    assert all(mc_sess.prev_weights_p == price_weights)
    # calculate possible_p & epsilon and check if is as expected:
    possible_p = np.linspace(start=mc_sess.b_min,
                             stop=mc_sess.b_max,
                             num=mc_sess.n_price_steps)
    epsilon = possible_p[1] - possible_p[0]
    assert isinstance(mc_sess.possible_p, np.ndarray)
    assert isinstance(mc_sess.epsilon, np.float64)
    assert all(mc_sess.possible_p == possible_p)
    assert mc_sess.epsilon == epsilon
