from ...common import create_user_resource_db


def test_load_sellers_class(session_class):
    resource_db = create_user_resource_db(nr_users=1, nr_resources_per_user=3)
    session_class.load_users_resources(users_resources=resource_db)

    for i, resource in enumerate(resource_db):
        seller_class = session_class.sellers_data[i]
        seller_class.user_id = resource["user"]
        seller_class.resource_id = resource["id"]

        # seller class for each resource should be init with 0 amount and
        # no measurements:
        assert seller_class.has_to_receive == 0
        assert seller_class.y is None
