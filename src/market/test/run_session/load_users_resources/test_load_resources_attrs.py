from ...common import create_user_resource_db


def test_load_resources_3u_3r(session_class):
    resource_db = create_user_resource_db(nr_users=3, nr_resources_per_user=3)
    assert isinstance(session_class.users_resources, list)
    assert len(session_class.users_resources) == 0
    session_class.load_users_resources(users_resources=resource_db)
    assert len(session_class.users_resources) == 9
    assert len(session_class.sellers_data) == 9
    assert len(session_class.users_list) == 3
    assert len(session_class.users_data) == 3


def test_load_resources_3u_2r(session_class):
    resource_db = create_user_resource_db(nr_users=3, nr_resources_per_user=2)
    assert isinstance(session_class.users_resources, list)
    assert len(session_class.users_resources) == 0
    session_class.load_users_resources(users_resources=resource_db)
    assert len(session_class.users_resources) == 6
    assert len(session_class.sellers_data) == 6
    assert len(session_class.users_list) == 3
    assert len(session_class.users_data) == 3


def test_load_resources_1u_3r(session_class):
    resource_db = create_user_resource_db(nr_users=1, nr_resources_per_user=3)
    assert isinstance(session_class.users_resources, list)
    assert len(session_class.users_resources) == 0
    session_class.load_users_resources(users_resources=resource_db)
    assert len(session_class.users_resources) == 3
    assert len(session_class.sellers_data) == 3
    assert len(session_class.users_list) == 1
    assert len(session_class.users_data) == 1
