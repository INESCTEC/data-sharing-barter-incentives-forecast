import pytest
from ..common import (
    create_buyer_bid_per_resource,
    create_user_resource_db
)

from src.market.helpers.db_helpers import get_measurements_data_mock


@pytest.fixture(scope="function")
def market_session_3u_3r(session_class):
    # Prepare market data:
    resource_db = create_user_resource_db(nr_users=3, nr_resources_per_user=3)
    bid_db = create_buyer_bid_per_resource(resource_db=resource_db)
    measurements_db = get_measurements_data_mock(
        users_resources=resource_db,
        market_launch_time=session_class.launch_time
    )

    session_class.load_users_resources(users_resources=resource_db)
    session_class.load_resources_bids(bids=bid_db)
    session_class.load_resources_measurements(measurements=measurements_db)
    session_class.start_session(api_controller=None)
    return session_class


@pytest.fixture(scope="function")
def market_session_2u_1r(session_class):
    # Prepare market data:
    resource_db = create_user_resource_db(nr_users=2, nr_resources_per_user=1)
    bid_db = create_buyer_bid_per_resource(resource_db=resource_db)
    measurements_db = get_measurements_data_mock(
        users_resources=resource_db,
        market_launch_time=session_class.launch_time
    )
    session_class.load_users_resources(users_resources=resource_db)
    session_class.load_resources_bids(bids=bid_db)
    session_class.load_resources_measurements(measurements=measurements_db)
    session_class.start_session(api_controller=None)
    return session_class


@pytest.fixture(scope="function")
def market_session_2u_2r(session_class):
    # Prepare market data:
    resource_db = create_user_resource_db(nr_users=2, nr_resources_per_user=2)
    bid_db = create_buyer_bid_per_resource(resource_db=resource_db)
    measurements_db = get_measurements_data_mock(
        users_resources=resource_db,
        market_launch_time=session_class.launch_time
    )
    session_class.load_users_resources(users_resources=resource_db)
    session_class.load_resources_bids(bids=bid_db)
    session_class.load_resources_measurements(measurements=measurements_db)
    session_class.start_session(api_controller=None)
    return session_class


@pytest.fixture(scope="function")
def market_session_3u_4r(session_class):
    # Prepare market data:
    resource_db = create_user_resource_db(nr_users=3, nr_resources_per_user=4)
    bid_db = create_buyer_bid_per_resource(resource_db=resource_db)
    measurements_db = get_measurements_data_mock(
        users_resources=resource_db,
        market_launch_time=session_class.launch_time
    )
    session_class.load_users_resources(users_resources=resource_db)
    session_class.load_resources_bids(bids=bid_db)
    session_class.load_resources_measurements(measurements=measurements_db)
    session_class.start_session(api_controller=None)
    return session_class
