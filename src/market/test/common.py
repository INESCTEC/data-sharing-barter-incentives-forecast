import numpy as np
from copy import deepcopy

from src.market import MarketClass
from src.market.test.database import (
    session_price_weights_template,
    session_data_template,
    buyer_bid_template,
    user_resources_template
)
from src.market.util.mock import (
    MeasurementsGenerator
)


def create_session_price_weights():
    # Prepare weights
    weights_data = session_price_weights_template()
    weights_ = [(x["id"], x["weights_p"]) for x in weights_data]
    sorted_weights_ = sorted(weights_, key=lambda tup: tup[1])
    return np.array([x[1] for x in sorted_weights_])


def create_market_session_data(
        convert_to_miota=False,
        use_custom_data=False,
        **kwargs,
):
    session_data = session_data_template()

    if use_custom_data:
        # update session data based on kwargs. If specific key doesnt exist,
        # default to template value.
        for key in session_data.keys():
            session_data[key] = kwargs.get(key, session_data[key])

    if convert_to_miota:
        converter = lambda x: np.float64(x / 1000000.0)
        session_data = deepcopy(session_data)
        session_data["b_min"] = converter(session_data["b_min"])
        session_data["b_max"] = converter(session_data["b_max"])
        session_data["market_price"] = converter(session_data["market_price"])

    return session_data


def create_user_resource(use_custom_data=False, **kwargs):
    user_resource = user_resources_template()
    if use_custom_data:
        # update bid data based on kwargs. If specific key doesnt exist,
        # default to template value.
        for key in user_resource.keys():
            user_resource[key] = kwargs.get(key, user_resource[key])
    return user_resource


def create_user_resource_db(nr_users, nr_resources_per_user):
    resource_id = 0
    resource_db = []
    for user_id in range(nr_users):
        for _ in range(nr_resources_per_user):
            resource_db.append(create_user_resource(use_custom_data=True,
                                                    user=str(user_id),
                                                    id=str(resource_id)))
            resource_id += 1
    return resource_db


def create_buyer_bid(convert_to_miota=False, use_custom_data=False, **kwargs):
    bid_data = buyer_bid_template()

    if use_custom_data:
        # update bid data based on kwargs. If specific key doesnt exist,
        # default to template value.
        for key in bid_data.keys():
            bid_data[key] = kwargs.get(key, bid_data[key])

    if convert_to_miota:
        converter = lambda x: np.float64(x / 1000000.0)
        bid_data = deepcopy(bid_data)
        bid_data["bid_price"] = converter(bid_data["bid_price"])
        bid_data["max_payment"] = converter(bid_data["max_payment"])

    bid_data["features_list"] = []

    return bid_data


def create_buyer_bid_per_resource(resource_db, convert_to_miota=True):
    bids = []
    bid_id = 0
    for resource in resource_db:
        resource_id = resource["id"]
        user_id = resource["user"]
        bids.append(create_buyer_bid(convert_to_miota=convert_to_miota,
                                     use_custom_data=True,
                                     user=user_id,
                                     resource=resource_id,
                                     id=bid_id))
        bid_id += 1
    return bids


def init_market_class(launch_time, session_data, price_weights_data):
    # Create market class obj:
    mc = MarketClass(n_jobs=1)
    mc.activate_debug_mode()
    mc.init_session(
        session_data=session_data,
        price_weights=price_weights_data,
        launch_time=launch_time
    )
    return mc
