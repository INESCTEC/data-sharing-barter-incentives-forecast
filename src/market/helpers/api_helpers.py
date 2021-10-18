import pandas as pd

from loguru import logger

# #############################################################################
# Get Session Data:
# #############################################################################


def get_session_data(api_controller):
    logger.info("Fetching session data ...")
    # Get current last active session (status='closed'):
    session_data = api_controller.list_last_session(status='closed')
    # Get session_id & fetch data for that session:
    active_session_id = session_data["market_session_id"]
    logger.debug(f"Retrieving data for Session ID {active_session_id}")
    # -- Session buyers bids:
    buyers_bids = api_controller.list_session_bids(active_session_id)
    # -- Session active sellers:
    active_sellers = api_controller.list_active_sellers()
    # -- Session weights:
    price_weights = api_controller.list_session_weights(active_session_id)
    logger.info("Fetching session data ... Ok!")
    return session_data, buyers_bids, active_sellers, price_weights


# #############################################################################
# Get Measurements Data for Session Agents:
# #############################################################################


def get_measurements_data_mock(api_controller, buyers_ids, sellers_ids,
                               market_launch_time):
    # todo: Right now working with mock data - replace by DB queries:
    from ..util.mock import MeasurementsGenerator
    logger.info("Querying agents measurements ...")
    agent_list = set(buyers_ids + sellers_ids)
    # Create fictitious measurements data
    mg = MeasurementsGenerator()
    measurements = {}
    for agent in sorted(agent_list):
        measurements[agent] = mg.generate_mock_data_sin(
            start_date=market_launch_time - pd.DateOffset(months=12),
            end_date=market_launch_time,
        )
    logger.info("Querying agents measurements ... Ok!")
    return measurements


def get_measurements_data(api_controller, buyers_ids, sellers_ids,
                          market_launch_time):
    # todo: implement this once data in DB
    raise NotImplementedError("Method not yet implemented.")


# #############################################################################
# Function to terminate sessions without bids:
# #############################################################################


def close_no_bids_session(api_controller,
                          curr_session_data,
                          curr_price_weights):
    import datetime as dt
    # -- Current session info:
    curr_session_id = curr_session_data["market_session_id"]
    curr_session_close_date = dt.datetime.strptime(
        curr_session_data["close_ts"],
        "%Y-%m-%dT%H:%M:%S.%fZ"
    ).date()
    # -- Next session info:
    next_session_ts = dt.datetime.utcnow()
    next_session_date = dt.datetime.utcnow().date()
    next_session_id = curr_session_id + 1
    if next_session_date > curr_session_close_date:
        next_session_number = 0
    else:
        next_session_number = curr_session_data["session_number"] + 1
    # -- Update market session status (set to "finished"):
    api_controller.update_market_session(
        session_id=curr_session_id,
        status="finished",
        finish_ts=next_session_ts
    )
    # -- Open new market session:
    api_controller.create_market_session(
        session_number=next_session_number,
        market_price=curr_session_data["market_price"],
        b_min=curr_session_data["b_min"],
        b_max=curr_session_data["b_max"],
        n_price_steps=curr_session_data["n_price_steps"],
        delta=curr_session_data["delta"]
    )
    # -- Post weights to new market session:
    api_controller.post_session_weights(
        session_id=next_session_id,
        weights_p=curr_price_weights
    )
