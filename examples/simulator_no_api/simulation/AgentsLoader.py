import os
import json
import datetime as dt

import pandas as pd

from loguru import logger


class AgentsLoader:
    """
    AgentsLoader Class responsible for:
    - Reading CSV data

    """
    def __init__(self, launch_time, market_session, data_path, bids_scenario,
                 delimiter=',', datetime_fmt="%Y-%m-%d %H:%M"):
        self.launch_time = launch_time
        self.market_session = market_session
        self.data_path = None
        self.users_resources = None
        self.users_features = None
        self.measurements = {}
        self.features = {}
        self.users_list = None
        self.resource_list = None
        self.measurements_list = None
        self.features_list = None
        self.bids_per_resource = None
        self.data_path = data_path
        self.bids_scenario = bids_scenario
        self.datetime_fmt = datetime_fmt
        self.delimiter = delimiter

    def read_dataset(self, data_type: str):
        """
        Read CSV data. Drops duplicates based on datetime and initializes
         a 'self.dataset' class attribute containing the loaded timeseries

        :param data_type:
        :param sep:
        :return:
        """
        if data_type not in ["measurements", "features"]:
            raise ValueError("data_type must be either "
                             "'measurements' or 'features'")

        # dataset path:
        dataset_path = os.path.join(self.data_path, f"{data_type}.csv")
        if os.path.exists(dataset_path):
            dataset = pd.read_csv(dataset_path, sep=self.delimiter)
            dataset.drop_duplicates("datetime", inplace=True)
            dataset.loc[:, 'datetime'] = pd.to_datetime(
                dataset["datetime"],
                format=self.datetime_fmt).dt.tz_localize("UTC")
            dataset.set_index("datetime", inplace=True)
            dataset = dataset.resample("h").mean()
            dataset.dropna(how="all", inplace=True)
        else:
            logger.warning(f"File {dataset_path} not found. "
                           f"Creating empty {data_type} dataset.")
            dataset = pd.DataFrame()
        return dataset

    def load_user_resources(self):
        """
        Loads user and user resources metadata.
        Initializes 'self.resource_list' class attribute with this information.
        """
        # user resources path for that dataset:
        user_res_path = os.path.join(self.data_path, "users_resources.json")
        with open(user_res_path, "r") as f:
            self.users_resources = json.load(f)

        self.users_list = [x["user"] for x in self.users_resources]
        self.resource_list = [x["id"] for x in self.users_resources]
        self.measurements_list = [x["id"] for x in self.users_resources if x["type"] == "measurements"]
        self.features_list = [x["id"] for x in self.users_resources if x["type"] == "features"]

        if len(set(self.resource_list)) != len(self.resource_list):
            raise AttributeError("There are repeated resource id's in user_resources.json.")

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
        dataset = self.read_dataset(data_type="measurements")

        if dataset.empty:
            exit("Empty measurements dataset. Cannot continue.")

        # make sure we only load data until market launch (historical)
        _ts = dataset[:end_date].index

        for resource_id in self.measurements_list:
            _v = dataset.loc[:end_date, f"{resource_id}"].dropna()
            _ts = _v.index
            _v = _v.values
            self.measurements[resource_id] = pd.DataFrame({
                "datetime": _ts,
                "value": _v,
                "variable": ["measurements"] * len(_ts),
                "units": ["w"] * len(_ts),
            }).set_index("datetime")
        return self

    def load_features(self):
        self.features = {}
        end_date = (self.launch_time + pd.DateOffset(days=3)).strftime("%Y-%m-%d %H:%M:%S.%f")
        dataset = self.read_dataset(data_type="features")

        for feature_id in self.features_list:
            feature_df = dataset.loc[:end_date, f"{feature_id}"].dropna()
            if feature_df.empty:
                self.features[feature_id] = pd.DataFrame({
                    "datetime": [],
                    "value": [],
                    "variable": []
                }).set_index("datetime")
            else:
                _ts = feature_df.index
                _v = feature_df.values
                self.features[feature_id] = pd.DataFrame({
                    "datetime": _ts,
                    "value": _v,
                    "variable": [f"feature_{feature_id}"] * len(_ts)
                }).set_index("datetime")
        return self

    def load_datasets(self):
        # Load user resources (metadata)
        self.load_user_resources()
        # load pre-defined bids
        self.load_bids(scenario=self.bids_scenario)
        # Read measurements data and assign to each user resource
        self.load_measurements()
        # Load features data and assign to each user resource
        self.load_features()

        return self
