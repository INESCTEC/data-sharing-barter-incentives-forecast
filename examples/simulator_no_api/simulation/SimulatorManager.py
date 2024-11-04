import os
import pandas as pd
import datetime as dt

from copy import deepcopy
from loguru import logger


__ROOT_PATH__ = os.path.dirname(__file__)


class SimulatorManager:
    """
    SimulatorManager class responsible for:
    - Creating the sessions list
    - Creating the reports directory
    - Creating the logger
    - Creating the report templates
    """

    def __init__(self,
                 dataset_path,
                 bids_scenario,
                 nr_sessions,
                 first_lt_utc,
                 session_freq,
                 delimiter=",",
                 datetime_fmt="%Y-%m-%d %H:%M",
                 price_up_data_path=None,
                 price_down_data_path=None,
                 price_spot_data_path=None,
                 agents_area_map_path=None,
                 report_name_suffix=None,
                 auto_feature_selection=False,
                 auto_feature_engineering=False,
                 ):

        # Simulator params:
        self.market_price = None
        self.market_price_weights = None
        self.nr_sessions = nr_sessions
        self.session_freq = session_freq
        self.DATASET_PATH = dataset_path
        self.DATASET_NAME = os.path.basename(self.DATASET_PATH)
        self.BIDS_SCENARIO = bids_scenario
        self.PRICE_UP_DATA_PATH = price_up_data_path
        self.PRICE_DOWN_DATA_PATH = price_down_data_path
        self.PRICE_SPOT_DATA_PATH = price_spot_data_path
        self.AGENTS_AREA_MAP_PATH = agents_area_map_path
        self.DATETIME_FMT = datetime_fmt
        self.DATA_DELIMITER = delimiter
        self.AUTO_FEATURE_SELECTION = auto_feature_selection
        self.AUTO_FEATURE_ENGINEERING = auto_feature_engineering
        self.REPORT_NAME_SUFFIX = report_name_suffix

        # parse first launch time:
        self.first_lt_utc = dt.datetime.strptime(first_lt_utc, "%Y-%m-%dT%H:%M:%SZ") # noqa

        # Create auxiliary variables:
        self.__create_reports_dir()
        self.__create_logger()
        self.__create_sessions_list()
        self.__create_report_templates()

    def __create_reports_dir(self):
        current_time = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        _d = os.path.dirname
        _fs = "autofs" if self.AUTO_FEATURE_SELECTION else "noautofs"
        report_dirname = f"{current_time}_{_fs}"

        if self.REPORT_NAME_SUFFIX is not None:
            report_dirname = f"{report_dirname}_{self.REPORT_NAME_SUFFIX}"

        self.REPORTS_PATH = os.path.join(_d(_d(__file__)),
                                         "files", "reports",
                                         self.DATASET_NAME,
                                         report_dirname)
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
        nr_sessions = self.nr_sessions
        self.SESSIONS_LAUNCH_TIME = pd.date_range(
            start=self.first_lt_utc,
            end=self.first_lt_utc + pd.DateOffset(
                hours=self.session_freq * nr_sessions
            ),
            freq="h"
        )
        self.SESSIONS_LIST = [(i, lt) for i, lt in enumerate(self.SESSIONS_LAUNCH_TIME)]  # noqa

    def __create_report_templates(self):
        # Template for market report:
        self.market_session_report = dict([(k, {}) for k, _ in self.SESSIONS_LIST])

        # Expected columns for CSV reports:
        self.session_details_fields = ['session_id',
                                       'session_lt',
                                       'market_price',
                                       'next_market_price',
                                       'identifier',
                                       'date',
                                       'status',
                                       'launch_ts',
                                       'elapsed_time',
                                       'next_weights_p',
                                       'prev_weights_p']
        self.buyers_results_fields = ['session_id',
                                      'user_id',
                                      'resource_id',
                                      'gain_func',
                                      'gain',
                                      'initial_bid',
                                      'final_bid',
                                      'max_payment',
                                      'has_to_pay']
        self.sellers_results_fields = ['session_id',
                                       'user_id',
                                       'resource_id',
                                       'has_to_receive',
                                       'shapley_value']

    def add_session_reports(self,
                            session_id,
                            session_lt,
                            session_details,
                            session_buyers_results,
                            session_buyers_forecasts,
                            session_sellers_results):
        # Session report template:
        sess_dict = {
            "session_details": [],
            "buyers_results": [],
            "sellers_results": [],
            "buyers_forecasts": [],
        }
        # Update Market dataframe with this session details:
        details_ = dict([(k, v) for k, v in session_details.items()])  # noqa
        # Add session details to market report:
        details_["session_id"] = session_id
        details_["session_lt"] = session_lt
        sess_dict["session_details"] = details_
        # Add buyers results to market report:
        sess_dict["buyers_results"] = list(session_buyers_results.values())
        # Add sellers to market report:
        sess_dict["sellers_results"] = list(session_sellers_results.values())
        # Add buyers forecasts:
        for k, v in session_buyers_forecasts.items():
            v_ = [{**d, "session_id": session_id} for d in v]
            sess_dict["buyers_forecasts"] += v_

        # Update market report with this session details:
        self.market_session_report[session_id] = sess_dict

    def reports_to_csv(self, sess_elapsed_time):
        data_details = []
        data_buyers = []
        data_sellers = []
        data_forecasts = []

        for session_id, session_data in self.market_session_report.items():
            if session_data == {}:
                # Skip empty sessions:
                continue
            # Session details:
            session_details = session_data["session_details"]
            session_details["elapsed_time"] = sess_elapsed_time
            data_details.append(session_details)
            # Buyers results:
            buyers_results = deepcopy(session_data["buyers_results"])
            buyers_results = [{**d, "session_id": session_id} for d in buyers_results]  # noqa
            data_buyers += buyers_results
            # Sellers results:
            sellers_results = deepcopy(session_data["sellers_results"])
            sellers_results = [{**d, "session_id": session_id} for d in sellers_results]  # noqa
            data_sellers += sellers_results
            # Buyers forecasts:
            buyers_forecasts = deepcopy(session_data["buyers_forecasts"])
            data_forecasts += buyers_forecasts

        # to CSV:
        pd.DataFrame(data_details, columns=self.session_details_fields).to_csv(os.path.join(self.REPORTS_PATH, "market.csv"), index=False)  # noqa
        pd.DataFrame(data_buyers, columns=self.buyers_results_fields).to_csv(os.path.join(self.REPORTS_PATH, "buyers.csv"), index=False)   # noqa
        pd.DataFrame(data_sellers, columns=self.sellers_results_fields).to_csv(os.path.join(self.REPORTS_PATH, "sellers.csv"), index=False) # noqa
        pd.DataFrame(data_forecasts).to_csv(os.path.join(self.REPORTS_PATH, "forecasts.csv"), index=False) # noqa
