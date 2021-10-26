import os
import json
import pytest
import numpy as np
import pandas as pd

from src.market.util.mock import (
    AgentsGenerator,
    SessionGenerator,
    MeasurementsGenerator
)

__TEST_DATA_DIR__ = os.path.join(os.path.dirname(__file__), "files")


@pytest.fixture
def init_configs():
    lt_ = '2020-05-01 10:00:00'
    lt_ = pd.to_datetime(lt_).tz_localize("UTC").to_pydatetime()
    return {
        "session_id": 1,
        "market_launch_time": lt_
    }


@pytest.fixture
def market_session(init_configs):
    # #########################################
    # Create Mock Data Session
    # #########################################
    sg = SessionGenerator()
    sg.create_session(
        session_id=init_configs["session_id"],
        launch_time=init_configs["market_launch_time"]
    )
    return sg


@pytest.fixture
def bids_3b_3s(init_configs, market_session):
    # ###################################################
    # Create Mock Data Agents:
    # ###################################################
    NR_BUYERS = 3
    NR_SELLERS = 3

    # Create fictitious bids:
    market_bid_id = 1
    ag = AgentsGenerator(launch_time=init_configs["market_launch_time"])
    for i in range(NR_BUYERS):
        ag.add_buyer(user=i,
                     market_price=market_session.market_price,
                     bid_price=np.float64(5.0),
                     max_payment=np.float64(1000),
                     market_bid_id=market_bid_id
                     )
        market_bid_id += 1
    for i in range(NR_SELLERS):
        ag.add_seller(user=i)
    return ag.buyers_bids, ag.sellers_list


@pytest.fixture
def measurements_bids_3b_3s(init_configs, bids_3b_3s):
    buyers_bids, sellers_ids = bids_3b_3s
    # Create fictitious measurements data
    buyers_ids = [x["user"] for x in buyers_bids]
    agent_list = set(buyers_ids + sellers_ids)
    mg = MeasurementsGenerator()
    measurements = {}

    for agent in sorted(agent_list):
        measurements[agent] = mg.generate_mock_data_sin(
            start_date=init_configs["market_launch_time"] - pd.DateOffset(
                years=1),  # noqa
            end_date=init_configs["market_launch_time"],
        )
        measurements[agent].set_index("datetime", inplace=True)
    return measurements


@pytest.fixture
def session_data_json():
    json_path = os.path.join(__TEST_DATA_DIR__,
                             "rest_api_response_examples",
                             "session_data_response.json")
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


@pytest.fixture
def buyers_bids_json():
    json_path = os.path.join(__TEST_DATA_DIR__,
                             "rest_api_response_examples",
                             "buyers_bids_response.json")
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data
