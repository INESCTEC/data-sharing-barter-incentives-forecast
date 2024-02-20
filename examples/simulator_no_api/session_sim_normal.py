# flake8: noqa
import gc
import sys

from time import time
from copy import deepcopy
from loguru import logger

# -- If needed to run via command line, add root proj to sys path:
# sys.path.append(r"<path_to_project>/data-sharing-barter-incentives-forecast")
from src.market import MarketClass
from src.market.helpers.units_helpers import (
    convert_session_data_to_mi,
    convert_buyers_bids_to_mi,
)

from simulation import SessionGenerator, AgentsLoader, SimulatorManager


if __name__ == '__main__':

    # -- Setup logger (removes existing logger + adds new sys logger):
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    # Set base simulation parameters:
    N_JOBS = 1
    simulation_params = {
        "dataset_path": "files/datasets/example_1",
        "report_name_suffix": "spearman",
        "auto_feature_selection": True,
        "auto_feature_engineering": True,
        "bids_scenario": "scenario_1",
        "nr_sessions": 10,
        "first_lt_utc": "2021-01-10T00:00:00Z",
        "session_freq": 1,
        "datetime_fmt": "%Y-%m-%d %H:%M",
        "delimiter": ","
    }

    # Load Session Configs:
    manager = SimulatorManager(**simulation_params)

    # -- Initialize helper variables:
    CURRENT_MARKET_PRICE = None
    CURRENT_PRICE_WEIGHTS = None

    # -- Run market sessions:
    for session_id, market_lt in manager.SESSIONS_LIST:
        logger.info("/" * 79)
        logger.info("\\" * 79)
        market_lt = market_lt.to_pydatetime()
        general_t0 = time()

        # #########################################
        # Create Mock Data Session
        # #########################################
        sg = SessionGenerator()
        if session_id > 0:
            # If not first session, update market price and weights based
            # on previous session results:
            sg.set_market_price(market_price=CURRENT_MARKET_PRICE * 1e6)
            sg.set_price_weights(price_weights=CURRENT_PRICE_WEIGHTS)

        # Create session:
        sg.create_session(session_id=session_id, launch_time=market_lt)

        # ###################################################
        # Create Mock Data Agents:
        # ###################################################
        # Create fictitious bids:
        ag = AgentsLoader(
            launch_time=market_lt,
            market_session=session_id,
            data_path=manager.DATASET_PATH,
            bids_scenario=manager.BIDS_SCENARIO,
            datetime_fmt=manager.DATETIME_FMT,
            delimiter=manager.DATA_DELIMITER
        ).load_datasets()

        # Session data:
        measurements = ag.measurements
        features = ag.features
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
        mc = MarketClass(n_jobs=N_JOBS,
                         auto_feature_engineering=manager.AUTO_FEATURE_ENGINEERING,  # noqa
                         auto_feature_selection=manager.AUTO_FEATURE_SELECTION)
        # -- Initialize market session:
        mc.init_session(
            session_data=session_data,
            price_weights=price_weights,
            launch_time=market_lt
        )
        # -- Display session details:
        mc.show_session_details()
        # -- Load resources bids:
        mc.load_users_resources(users_resources=users_resources)
        mc.load_resources_bids(bids=bids_per_resource)
        # -- Load resources measurements data:
        mc.load_resources_measurements(measurements=measurements)
        mc.load_resources_features(features=features)
        # -- Run market session:
        mc.define_payments_and_forecasts()
        # -- Display session results
        mc.define_sellers_revenue()
        # -- Save & validate session results (raise exception if not valid)
        mc.save_session_results(save_forecasts=True)
        mc.validate_session_results(raise_exception=True)
        # -- Display session results
        mc.show_session_results()
        # -- Update market price for next session:
        mc.update_market_price()
        # -- Display session results
        mc.show_session_results()

        #################################
        # Finalize Session
        #################################
        manager.add_session_reports(
            session_id=session_id,
            session_lt=market_lt,
            session_details=deepcopy(mc.mkt_sess.details),
            session_buyers_results=deepcopy(mc.mkt_sess.buyers_results),
            session_buyers_forecasts=deepcopy(mc.mkt_sess.buyers_forecasts),
            session_sellers_results=deepcopy(mc.mkt_sess.sellers_results),
        )

        # Save reports to csv:
        elapsed_time = time() - general_t0
        manager.reports_to_csv(sess_elapsed_time=elapsed_time)

        # Update variables for next session
        CURRENT_MARKET_PRICE = mc.mkt_sess.next_market_price
        CURRENT_PRICE_WEIGHTS = mc.mkt_sess.next_weights_p

        # Display next session references
        logger.info(">" * 70)
        logger.info("Next session references:")
        logger.info(f"Market price: {CURRENT_MARKET_PRICE}")
        logger.info(f"Price weights: {CURRENT_PRICE_WEIGHTS}")
        logger.info("<" * 70)

        # Delete objects to free memory
        del mc
        del ag
        del sg
        gc.collect()

