# flake8: noqa
import datetime as dt

from loguru import logger

from data import SessionGenerator, AgentsLoader
from helpers.reporting import create_report
from src.market.helpers.units_helpers import (
    convert_session_data_to_mi,
    convert_buyers_bids_to_mi,
)


if __name__ == '__main__':
    from examples.simulator_no_api.SimulatorConfig import Config
    from src.market import MarketClass
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    N_JOBS = -1

    # Load Session Configs:
    cfg = Config(
        dataset_path="data/datasets/carla_paper",
        bids_scenario="scenario_1",
        nr_sessions=1000,
        first_lt_utc=dt.datetime(2020, 5, 1, 10, 00, 3, 4536),
        session_freq=1,
    )
    CURRENT_MARKET_PRICE = None
    CURRENT_PRICE_WEIGHTS = None
    SESSIONS_LIST = cfg.SESSIONS_LIST
    RESULTS = cfg.RESULTS
    BUYERS_DF = cfg.BUYERS_DF
    SELLERS_DF = cfg.SELLERS_DF
    MARKET_DF = cfg.MARKET_DF
    REPORTS_PATH = cfg.REPORTS_PATH
    DATASET_PATH = cfg.DATASET_PATH

    # -- Run market sessions:
    for session_id, market_lt in enumerate(cfg.SESSIONS_LIST):
        logger.info("/" * 79)
        logger.info("\\" * 79)
        market_lt = market_lt.to_pydatetime()

        # #########################################
        # Create Mock Data Session
        # #########################################
        sg = SessionGenerator()
        if session_id > 0:
            sg.set_market_price(market_price=CURRENT_MARKET_PRICE)
            sg.set_price_weights(price_weights=CURRENT_PRICE_WEIGHTS)

        # Create session:
        sg.create_session(session_id=session_id, launch_time=market_lt)

        # ###################################################
        # Create Mock Data Agents:
        # ###################################################
        # Create fictitious bids:
        ag = AgentsLoader(launch_time=market_lt, market_session=session_id)
        ag.read_data(path=DATASET_PATH)
        ag.load_user_resources()
        ag.load_bids(scenario=cfg.BIDS_SCENARIO)
        measurements = ag.load_measurements()

        # Session data:
        session_data = sg.session_data
        price_weights = sg.price_weights
        bids_per_resource = ag.bids_per_resource
        users_resources = ag.users_resources

        ###################################
        # Convert units from IOTA to MIOTA:
        # ####################################
        session_data = convert_session_data_to_mi(data=session_data)
        bids_per_resource = convert_buyers_bids_to_mi(bids=bids_per_resource)

        # ################################
        # Run Market Session
        # ################################
        mc = MarketClass(n_jobs=N_JOBS)
        mc.init_session(
            session_data=session_data,
            price_weights=price_weights,
            launch_time=market_lt
        )
        mc.show_session_details()
        # -- Load resources bids:
        mc.load_users_resources(users_resources=users_resources)
        mc.load_resources_bids(bids=bids_per_resource)
        # -- Load resources measurements data:
        mc.load_resources_measurements(measurements=measurements)
        # -- Run market session:
        mc.define_payments_and_forecasts()
        mc.define_sellers_revenue()
        mc.save_session_results()
        mc.validate_session_results(raise_exception=True)
        # -- Display session results
        mc.show_session_results()
        # -- Update market price for next session:
        mc.update_market_price()
        # -- Display session results
        mc.show_session_results()

        # Save session results
        RESULTS[session_id] = mc.mkt_sess
        BUYERS_DF, SELLERS_DF, MARKET_DF = create_report(
            session_id=session_id,
            session_lt=market_lt,
            buyers_df=BUYERS_DF,
            sellers_df=SELLERS_DF,
            market_df=MARKET_DF,
            market_sess=mc.mkt_sess,
            path=REPORTS_PATH,
        )

        # Update variables for next session
        CURRENT_MARKET_PRICE = mc.mkt_sess.next_market_price
        CURRENT_PRICE_WEIGHTS = mc.mkt_sess.next_weights_p

        logger.info(">" * 70)
        logger.info("Next session references:")
        logger.info(f"Market price: {CURRENT_MARKET_PRICE}")
        logger.info(f"Price weights: {CURRENT_PRICE_WEIGHTS}")
        logger.info("<" * 70)
