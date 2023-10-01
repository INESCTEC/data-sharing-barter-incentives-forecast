import os
import json
import datetime as dt

import pandas as pd


class AgentsLoader:
    """
    AgentsLoader Class responsible for:
    - Reading CSV data

    """
    def __init__(self, launch_time, market_session, data_path, bids_scenario):
        self.launch_time = launch_time
        self.market_session = market_session
        self.data_path = None
        self.dataset = None
        self.users_resources = None
        self.measurements = {}
        self.resource_list = None
        self.bids_per_resource = None
        self.data_path = data_path
        self.bids_scenario = bids_scenario

    def read_data(self, path: str, sep: str = ','):
        """
        Read CSV data. Drops duplicates based on datetime and initializes
         a 'self.dataset' class attribute containing the loaded timeseries

        :param path:
        :param sep:
        :return:
        """
        self.data_path = path
        # dataset path:
        dataset_path = os.path.join(path, "dataset.csv")
        self.dataset = pd.read_csv(dataset_path, sep=sep)
        self.dataset.drop_duplicates("datetime", inplace=True)
        self.dataset.loc[:, 'datetime'] = pd.to_datetime(
            self.dataset["datetime"],
            format="%Y-%m-%d %H:%M").dt.tz_localize("UTC")
        self.dataset.set_index("datetime", inplace=True)
        return self

    def load_user_resources(self):
        """
        Loads user and user resources metadata.
        Initializes 'self.resource_list' class attribute with this information.
        """
        # user resources path for that dataset:
        user_res_path = os.path.join(self.data_path, "user_resources.json")
        with open(user_res_path, "r") as f:
            self.users_resources = json.load(f)
        self.resource_list = [x["id"] for x in self.users_resources]

        return self

    def load_bids(self, scenario: str):
        bids_path = os.path.join(self.data_path, "bids", scenario, "bids.json")
        with open(bids_path, "r") as f:
            self.bids_per_resource = json.load(f)
        self.__add_bid_extra_fields()

    def __add_bid_extra_fields(self):
        # Add other fields that exist in bids DB (but not considered in SIM)
        dt_now = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(len(self.bids_per_resource)):
            self.bids_per_resource[i]["id"] = i
            self.bids_per_resource[i]["confirmed"] = True
            self.bids_per_resource[i]["has_forecasts"] = False
            self.bids_per_resource[i]["market_session"] = self.market_session
            self.bids_per_resource[i]["registered_at"] = dt_now
            self.bids_per_resource[i]["tangle_msg_id"] = os.urandom(24)

    def load_measurements(self):
        self.measurements = {}
        end_date = self.launch_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        _ts = self.dataset[:end_date].index

        for resource_id in self.resource_list:
            _v = self.dataset.loc[:end_date, f"{resource_id}"].values
            self.measurements[resource_id] = pd.DataFrame({
                "datetime": _ts,
                "value": _v,
                "variable": ["measurements"] * len(_ts),
                "units": ["w"] * len(_ts),
            }).set_index("datetime")

        return self.measurements

    def load_datasets(self):
        self.read_data(path=self.data_path)  # Read csv files
        self.load_user_resources()  # Load user resources (metadata)
        self.load_bids(scenario=self.bids_scenario)  # load pre-defined bids
        # Read measurements data and assign to each user resource
        self.load_measurements()
        return self
