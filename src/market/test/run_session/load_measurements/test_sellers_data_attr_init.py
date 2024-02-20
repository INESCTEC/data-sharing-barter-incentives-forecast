from ...common import create_user_resource_db
from src.market.helpers.db_helpers import get_measurements_data_mock


# todo: add tests to load measurements w/ bids (and check buyers_data)

def test_load_measurements_no_bids(session_class, market_launch_time):
    resource_db = create_user_resource_db(nr_users=3, nr_resources_per_user=3)
    measurements = get_measurements_data_mock(
        users_resources=resource_db,
        market_launch_time=market_launch_time
    )
    # Load user resource database:
    session_class.load_users_resources(users_resources=resource_db)
    # Load user measurements database:
    session_class.load_resources_measurements(measurements=measurements)
    # Check if all selling resources have expected measurements:
    for resource in resource_db:
        resource_id = resource["id"]
        assert session_class.sellers_data[resource_id].y.shape == measurements[resource_id].shape
