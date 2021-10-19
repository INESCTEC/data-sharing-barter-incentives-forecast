# flake8: noqa
import os
import sys
import numpy as np
import pandas as pd

import datetime as dt

from loguru import logger


def report(session_id, session_lt, buyers_df, sellers_df, market_df, market_sess, path):
    sess_dict = {}
    for k, v in market_sess.details.items():
        sess_dict[k] = str(v)
    sess_dict["session_id"] = session_id
    sess_dict["session_lt"] = session_lt
    market_df = market_df.append(pd.DataFrame(sess_dict, index=[0]),
                                 ignore_index=True)
    for bb in market_sess.buyers_results.values():
        bb["session_id"] = session_id
        bb["session_lt"] = session_lt
        buyers_df = buyers_df.append(pd.DataFrame(bb, index=[0]),
                                     ignore_index=True)

    for ss in market_sess.sellers_results.values():
        ss["session_id"] = session_id
        ss["session_lt"] = market_lt
        sellers_df = sellers_df.append(pd.DataFrame(ss, index=[0]),
                                       ignore_index=True)

    # Update report files:
    market_df.to_csv(os.path.join(path, "market.csv"), index=False)
    buyers_df.to_csv(os.path.join(path, "buyers.csv"), index=False)
    sellers_df.to_csv(os.path.join(path, "sellers.csv"), index=False)
    return buyers_df, sellers_df, market_df


if __name__ == '__main__':
    from examples.simulator_no_api.SimulatorConfig import Config
    from src.market import MarketClass
    from src.market.util.mock import AgentsGenerator, SessionGenerator

    # Load Session Configs:
    cfg = Config(
        dataset_path="data/mock_measurements.csv",
        nr_sessions=1000,
        first_lt_utc=dt.datetime(2020, 5, 1, 10, 00, 3, 4536),
        session_freq=1,
        n_buyers=3,
        n_sellers=0,
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
    BUYER_AGENTS = cfg.BUYER_AGENTS
    SELLER_AGENTS = cfg.SELLER_AGENTS
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
        active_session = sg.active_session
        price_weights = sg.price_weights

        # ###################################################
        # Create Mock Data Agents:
        # ###################################################
        # Create fictitious bids:
        ag = AgentsGenerator(launch_time=market_lt)
        ag.read_mock_dataset(path=DATASET_PATH)
        for a in BUYER_AGENTS:
            ag.add_buyer(user=a,
                         market_price=sg.market_price,
                         bid_price=np.float64(5),
                         max_payment=np.float64(1000))
            # Remember - A buyer is also a seller (always)
            ag.add_seller(user=a)
        for a in SELLER_AGENTS:
            ag.add_seller(user=a)

        buyers_bids = ag.buyers_bids
        active_sellers = ag.sellers_list

        # Create fictitious measurements data
        buyers_ids = [x["user"] for x in buyers_bids]
        sellers_ids = active_sellers
        agent_list = set(buyers_ids + sellers_ids)
        measurements = {}
        for i, agent in enumerate(sorted(agent_list)):
            measurements[agent] = ag.get_measurements(
                agent_id=i,
                end_date=market_lt
            )

        # ################################
        # Create & Run Market Session
        # ################################
        mc = MarketClass()
        mc.init_session(
            session_data=active_session,
            price_weights=price_weights,
            launch_time=market_lt
        )
        mc.start_session()
        mc.show_session_details()
        # -- Load agents bids:
        mc.load_buyers_bids(bids=buyers_bids)
        mc.load_sellers(identifiers=active_sellers)
        # -- Load agents measurements data:
        mc.load_agents_measurements(measurements=measurements)
        # -- Run market session:
        mc.run_session()
        # -- Display session results
        mc.show_session_results()
        # -- Update market price for next session:
        mc.update_market_price()
        # -- End session:
        mc.end_session()

        # Save session results
        RESULTS[session_id] = mc.mkt_sess
        buyers_df, sellers_df, market_df = report(
            session_id=session_id,
            session_lt=market_lt,
            buyers_df=BUYERS_DF,
            sellers_df=SELLERS_DF,
            market_df=MARKET_DF,
            market_sess=mc.mkt_sess,
            path=REPORTS_PATH
        )

        # Update variables for next session
        CURRENT_MARKET_PRICE = mc.mkt_sess.next_market_price
        CURRENT_PRICE_WEIGHTS = mc.mkt_sess.next_weights_p

        logger.info(">" * 70)
        logger.info("Next session references:")
        logger.info(f"Market price: {CURRENT_MARKET_PRICE}")
        logger.info(f"Price weights: {CURRENT_PRICE_WEIGHTS}")
        logger.info("<" * 70)
