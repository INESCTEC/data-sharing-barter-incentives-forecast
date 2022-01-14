import os
import pandas as pd
import datetime as dt

from loguru import logger


__ROOT_PATH__ = os.path.dirname(__file__)


class Config:

    def __init__(self,
                 dataset_path,
                 bids_scenario,
                 nr_sessions,
                 first_lt_utc,
                 session_freq,
                 price_up_data_path=None,
                 price_down_data_path=None,
                 price_spot_data_path=None,
                 agents_area_map_path=None,
                 ):
        self.nr_sessions = nr_sessions
        self.first_lt_utc = first_lt_utc
        self.session_freq = session_freq
        self.DATASET_PATH = dataset_path
        self.BIDS_SCENARIO = bids_scenario
        self.PRICE_UP_DATA_PATH = price_up_data_path
        self.PRICE_DOWN_DATA_PATH = price_down_data_path
        self.PRICE_SPOT_DATA_PATH = price_spot_data_path
        self.AGENTS_AREA_MAP_PATH = agents_area_map_path
        self.__create_reports_dir()
        self.__create_logger()
        self.__create_sessions_list()
        self.__create_sessions_reports()

    def __create_reports_dir(self):
        current_time = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.REPORTS_PATH = os.path.join(os.path.dirname(__file__),
                                         "reports",
                                         current_time)
        os.makedirs(self.REPORTS_PATH, exist_ok=True)

    def __create_logger(self):
        # logger:
        format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}"
        logger.add(os.path.join(self.REPORTS_PATH, "logfile.log"),
                   format=format,
                   level='INFO',
                   backtrace=True)
        logger.info("-" * 79)

    def __create_sessions_list(self):
        # Simulator params:
        nr_sessions = 1000
        self.SESSIONS_LIST = pd.date_range(
            start=self.first_lt_utc,
            end=self.first_lt_utc + pd.DateOffset(
                hours=self.session_freq * nr_sessions
            ),
            freq="H"
        )

    def __create_sessions_reports(self):
        self.RESULTS = {}
        self.BUYERS_DF = pd.DataFrame(
            columns=['session_id',
                     'session_lt',
                     'identifier',
                     'gain_func',
                     'gain',
                     'initial_bid',
                     'final_bid',
                     'max_payment',
                     'has_to_pay'])
        self.SELLERS_DF = pd.DataFrame(columns=['session_id',
                                                'session_lt',
                                                'identifier',
                                                'bid',
                                                'has_to_receive'])
        self.MARKET_DF = pd.DataFrame(columns=['session_id', 'session_lt',
                                               'market_price',
                                               'next_market_price',
                                               'identifier',
                                               'date',
                                               'status',
                                               'launch_ts',
                                               'next_weights_p',
                                               'prev_weights_p'])
