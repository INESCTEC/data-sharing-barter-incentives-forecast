import pandas as pd

from time import time
from conf import settings
from loguru import logger
from joblib import Parallel, delayed

# -- Helper funcs:
from src.market.helpers.market_helpers import (
    calc_buyer_payment,
    calc_sellers_revenue,
    market_price_update_parallel,
)

from src.market.helpers.db_helpers import (
    get_measurements_data,
    upload_forecasts,
    update_bid_has_forecast

)

# -- Market entities classes:
from src.market.UserClass import UserClass
from src.market.BuyerClass import BuyerClass
from src.market.SellerClass import SellerClass
from src.market.SessionClass import SessionClass
from src.market.util.custom_exceptions import (
    NoMarketDataException,
    NoMarketBuyersExceptions,
    NoMarketUsersExceptions
)

# -- Feature Selection
from src.market.preprocessing.feature_selection.feature_preprocess import FeatureProcess

# -- Feature Engineering:
from .preprocessing.feature_engineering.autocorrelation import autocorrelation_analysis  # noqa
from .preprocessing.feature_engineering.construct_inputs_funcs import construct_inputs_from_lags  # noqa

# -- Mock data imports:
from src.market.helpers.model_helpers import create_forecast
from src.market.helpers.units_helpers import convert_mi_to_i


class MarketClass:
    DEBUG = False
    N_HOURS = 24 * 14                  # no. hours in evaluation period
    FORECAST_HORIZON = settings.MARKET_FORECAST_HORIZON  # forecast horizon
    N_HOURS_IN_HIST = 8760             # no. hours in historical data
    MARKET_FEE_PCT = 0.05              # market fee applied to buyer payment
    REVENUE_K = 5
    REVENUE_LAMBDA = 1
    FORECASTS_TABLE = "market_forecasts"
    MEASUREMENTS_TABLE = "market_forecasts"
    BIDS_TABLE = "market_session_bid"
    MARKET_TZ = "utc"

    def __init__(self, n_jobs=-1,
                 enable_db_uploads=False,
                 auto_feature_engineering=True,
                 auto_feature_selection=True):
        self.users_data = {}
        self.users_list = []
        self.users_resources = []
        self.buyers_data = {}
        self.sellers_data = {}
        self.mkt_sess = None
        self.launch_time = None
        self.finish_time = None
        self.buyer_outputs = []
        self.n_jobs = n_jobs
        self.db_uploads = enable_db_uploads
        self.auto_feature_engineering = auto_feature_engineering
        self.auto_feature_selection = auto_feature_selection
        self.forecast_start = None
        self.forecast_end = None
        self.forecast_range = None

        if not self.auto_feature_engineering:
            logger.warning("Disabling auto_feature_engineering might affect "
                           "the number of features available in the market "
                           "dataset, for each buyer. To disable this, make "
                           "sure that you have sellers / buyers providing "
                           "their own 'features' (future available data) in "
                           "their participation in the market session.")

    def activate_debug_mode(self):
        self.DEBUG = True
        self.db_uploads = False
        logger.remove()

    def init_session(self, session_data, price_weights, launch_time):
        self.launch_time = launch_time
        self.mkt_sess = SessionClass(
            launch_ts=launch_time,
            session_id=session_data["id"],
            session_number=session_data["session_number"],
            session_date=session_data["session_date"],
            status=session_data["status"],
            market_price=session_data["market_price"],
            b_min=session_data["b_min"],
            b_max=session_data["b_max"],
            n_price_steps=session_data["n_price_steps"],
            delta=session_data["delta"]
        )
        self.mkt_sess.set_previous_price_weights(weights_p=price_weights)
        self.mkt_sess.validate_attributes()
        self.mkt_sess.set_initial_conditions()
        return self

    def start_session(self, api_controller=None):
        # todo: check api responses
        self.mkt_sess.start_session()
        if api_controller is not None:
            api_controller.update_market_session(
                session_id=self.mkt_sess.session_id,
                status=self.mkt_sess.status,
                launch_ts=self.mkt_sess.launch_ts
            )

    def end_session(self, api_controller=None):
        # todo: check api responses
        self.mkt_sess.end_session()
        if api_controller is not None:
            api_controller.update_market_session(
                session_id=self.mkt_sess.session_id,
                status=self.mkt_sess.status,
                finish_ts=self.mkt_sess.finish_ts
            )

    def show_session_details(self):
        if self.mkt_sess is None:
            exit("Error! Must init a session first!")
        logger.info("-" * 70)
        logger.info(">> Session details:")
        logger.info(f"Session ID: {self.mkt_sess.session_id}")
        logger.info(f"Session Number: {self.mkt_sess.session_number}")
        logger.info(f"Session Date: {self.mkt_sess.session_date}")
        logger.info(f"Session Launch Time: {self.mkt_sess.launch_ts}")
        logger.info(f"Market Price: {self.mkt_sess.market_price}Mi")
        logger.info(f"Current Price Weights: {self.mkt_sess.prev_weights_p}")

    def show_session_results(self):
        import json
        logger.info("-" * 70)
        logger.info(f">> Session {self.mkt_sess.session_id} results:")
        logger.info("-" * 70)
        logger.info(">>>> Per user & resource:")
        logger.info(f"Buyers:\n"
                    f"{json.dumps(self.mkt_sess.buyers_results, indent=2)}")
        logger.info("-")
        logger.info(f"Sellers:\n"
                    f"{json.dumps(self.mkt_sess.sellers_results, indent=2)}")
        logger.info("-" * 70)
        logger.info(">>>> General (aggregated view):")
        logger.info(f"Buyers:\n{json.dumps(self.mkt_sess.buyer_payment_per_user, indent=2)}")  # noqa
        logger.info("-")
        logger.info(f"Sellers:\n{json.dumps(self.mkt_sess.seller_revenue_per_user, indent=2)}")  # noqa
        logger.info("-")
        logger.info(f"Market (fees): \n"
                    f"Total: "
                    f"{self.mkt_sess.details['total_market_fee']}\n"
                    f"Per resource:\n"
                    f"{json.dumps(self.mkt_sess.details['market_fee_per_resource'], indent=2)}")  # noqa
        logger.info("-" * 70)

    def load_resources_bids(self, bids: list):
        if (not isinstance(bids, list)) or \
                (len(bids) > 0) and (not isinstance(bids[0], dict)):
            raise TypeError("Error! bids argument must be a list of dicts")

        for buyer_bid in bids:
            user_id = buyer_bid["user"]
            resource_id = buyer_bid["resource"]
            if user_id not in self.users_list:
                logger.warning(f"Unable to load bid for user/resource "
                               f"{user_id}/{resource_id}. This user resource "
                               f"was not properly loaded into user list.")
            else:
                # Init Buyer class with each bid information:
                self.buyers_data[resource_id] = BuyerClass(
                    user_id=user_id,
                    resource_id=resource_id,
                    initial_bid=buyer_bid["bid_price"],
                    max_payment=buyer_bid["max_payment"],
                    gain_func=buyer_bid["gain_func"],
                    market_bid_id=buyer_bid["id"],
                    features_list=buyer_bid.get("features_list", [])
                ).validate_attributes()

    def load_users_resources(self, users_resources: list):
        if not isinstance(users_resources, list):
            raise TypeError("Error! a list of resources must be provided")

        # Init Seller class with each seller identification:
        self.users_resources = users_resources
        logger.debug(f"\nUsers resources (to load):"
                     f"\n{users_resources}")

        # Load each user resource (measurements or features) data into
        # a sellers class:
        for resource_data in self.users_resources:
            user_id = resource_data["user"]
            resource_id = resource_data["id"]
            resource_type = resource_data["type"]
            self.sellers_data[resource_id] = SellerClass(
                user_id=user_id,
                resource_id=resource_id,
                resource_type=resource_type
            ).validate_attributes()

            if user_id not in self.users_list:
                self.users_list.append(user_id)

        # Load users data (based on resource id's)
        for user_id in self.users_list:
            # Compile list of features for this user:
            user_features_ = [x["id"] for x in self.users_resources if x["user"] == user_id and x["type"] == "features"]  # noqa
            self.users_data[user_id] = UserClass(
                user_id=user_id,
                user_features_list=user_features_
            )

        logger.debug(f"\nLoaded users data:\nusers_list{self.users_list}")

    def load_resources_measurements(self, measurements: dict):
        """
        Load measurements data into each agent class. Namely:
        - Buying agents resource measurements (forecast target datasets)
        - Selling agents resource measurements (to be used as forecast lags)

        :param measurements: Dictionary with measurements data for each
        agent resource identifier

        :return:
        """
        if not isinstance(measurements, dict):
            raise TypeError("Error! measurements arg. must be a dict")
        # Intersection - agents that are sellers & buyers
        buyers_resources_ = list(self.buyers_data.keys())
        # -- Only load measurements resources for selling agents:
        sellers_resources_ = [x[0] for x in self.sellers_data.items() if x[1].resource_type == "measurements"]  # noqa
        resource_list = set(buyers_resources_ + sellers_resources_)
        # Assign measurements data to each agent class:
        default_df = pd.DataFrame(columns=["datetime", "value"])
        for resource_id in sorted(resource_list):
            # Fetch agent data (empty dataset if key not found)
            _df = measurements.get(resource_id, default_df)
            if resource_id in self.buyers_data:
                self.buyers_data[resource_id].set_measurements(_df)
            if resource_id in self.sellers_data:
                self.sellers_data[resource_id].set_data(_df)
        return self

    def load_resources_features(self, features: dict):
        if not isinstance(features, dict):
            raise TypeError("Error! features arg. must be a dict")
        # In contrast to the load_resources_measurements method, here we only
        # load features data (aka explanatory variables shared by all agents)
        # to the market database
        resource_list = [x[0] for x in self.sellers_data.items() if x[1].resource_type == "features"]  # noqa
        # Assign measurements data to each agent class:
        default_df = pd.DataFrame(columns=["datetime", "value"])

        for resource_id in sorted(resource_list):
            # Fetch agent data (empty dataset if key not found)
            _df = features.get(resource_id, default_df)
            self.sellers_data[resource_id].set_data(_df)

        return self

    @staticmethod
    def __preprocess_buyer_data(data, expected_dates):
        """
        Resample data to hourly resolution
        Reindex so missing dates are market as NA

        :param data:
        :param expected_dates:
        :return:
        """
        data = data.resample('h').mean()
        data = data.reindex(expected_dates)
        return data

    def __create_market_dataset(self):
        """
        Create dataset with sellers data:

        1. Defines expected datetime range of market dataset
        2. Join sellers datasets
        3. Fill missing values (with zeros)

        :return: pd.DataFrame - sellers market dataset
        """
        # Define expected datetime range:
        logger.info("Creating market dataset ...")
        _end_date = self.launch_time.replace(minute=0, second=0, microsecond=0)
        _end_date = _end_date + pd.DateOffset(hours=self.FORECAST_HORIZON)
        _lookback_time = self.N_HOURS_IN_HIST - 1 + self.FORECAST_HORIZON
        _range = pd.date_range(
            start=_end_date - pd.DateOffset(hours=_lookback_time),
            end=_end_date,
            tz=self.MARKET_TZ,
            freq="h"
        )

        # Add sellers data:
        market_df = pd.DataFrame(index=_range)
        for seller_id, seller_cls in self.sellers_data.items():
            df_ = seller_cls.y[["value"]]
            df_ = df_.rename(columns={"value": seller_id})
            if not df_.empty:
                df_ = df_.resample("h").mean()
                market_df = market_df.join(df_, how="left")
            else:
                logger.warning(f"Empty dataset for resource {seller_id}")

        # Check if there is no market data:
        if market_df.dropna(how="all").empty:
            e_msg = "Error! No market dataset available. Terminating ..."
            logger.error(e_msg)
            raise NoMarketDataException(e_msg)
        else:
            logger.info("Creating market dataset ... Ok!")
            return market_df

    def __create_market_features(self, market_df: pd.DataFrame):
        """
        Create market features dataset. This dataset is created on market
        session start and will be used to complement each buyer information
        (i.e., buyer_x dataset) with remaining information in the market
        data pool (i.e., sellers dataset)

        If auto_feature_engineering is True, then the market features dataset
        will be created with lagged features for each seller 'measurements'
        resource. The 'lag' reference will be the difference (in hours) between
        the last date in the forecast horizon and the last date available in
        the dataset.

        'features' resources will be added to the feature dataset as is.

        Data imputation mechanisms::
        1. Backwards fill (limit 2h)
        2. Fill remaining NaN with zeros (this will penalize sellers with no
        data available for some hours)

        :param market_df: (pd.DataFrame) market dataset (sellers data)
        :return: (pd.DataFrame) market features dataset
        """
        logger.info("Creating market features ...")
        # go to market dataset and ignore agent_id measurements
        feat_df = pd.DataFrame(index=market_df.index)  # assure idx = buyer

        # For seller resource timeseries in market dataset:
        for seller_id in market_df.columns:
            type_ = self.sellers_data[seller_id].resource_type
            if type_ == "measurements":
                # -> create lagged features
                if self.auto_feature_engineering:
                    # Filter data for specific seller resource:
                    _dataset = market_df[[seller_id]]
                    # Get last available datetime:
                    _last_date = _dataset.dropna().index[-1]
                    # Get lag reference (difference (in hours) between the
                    # last date in forecast horizon and last available date
                    lag_ref = -int((self.forecast_range[-1] - _last_date).total_seconds() / 3600)  # noqa
                    # Create lag reference:
                    lag_vars = {seller_id: [('hour', [lag_ref])]}
                    lag_name_prefix = f"seller__{seller_id}__"
                    feat_df = construct_inputs_from_lags(
                        df=_dataset,
                        inputs=feat_df,
                        lag_vars=lag_vars,
                        lag_tz=self.MARKET_TZ,
                        infer_dst_lags=False,
                        lag_name_prefix=lag_name_prefix
                    )
            elif type_ == "features":
                # -> add to market features dataset as it is
                _name = f"seller__{seller_id}"
                feat_df.loc[:, _name] = market_df[seller_id]

        # Data imputation:
        feat_df.dropna(how="all", inplace=True)
        feat_df.bfill(limit=2, inplace=True)
        # Todo: find better imputation strategy (or drop NaN)
        feat_df.fillna(feat_df.mean(), inplace=True)
        feat_df = feat_df.asfreq('h')
        logger.info("Creating market features ... Ok!")
        return feat_df

    def __select_market_features(self,
                                 resource_id: str,
                                 user_features_list: list,
                                 market_x_full: pd.DataFrame):
        """
        Select all market features except the ones for a specif agent_id

        Removes user measurements (target) related features (e.g., lags)
        and user features (e.g., user_id, resource_id, etc.)
        Note: the user suggested features are already considered when
        creating the buyer_x dataset (see __create_buyer_features method)

        :param resource_id: Agent identifier
        :param user_features_list: List of all features sent by the agent to
         the market
        :param market_x_full: Market dataset
        :return:
        """

        cols_to_remove_ = [resource_id] + user_features_list
        _cols = [x for x in market_x_full.columns if x.split('__')[1] not in cols_to_remove_]  # noqa
        # Check which of the valid cols have no NaN in forecast horizon
        check_nulls = market_x_full[_cols].loc[self.forecast_range].isnull().any()  # noqa
        # get name (index) of columns with information available
        valid_cols = check_nulls[check_nulls == False].index
        return market_x_full[valid_cols]

    def __create_buyer_features(self,
                                buyer_y: pd.DataFrame,
                                target_resource_id: int,
                                suggested_features: list,
                                market_features: pd.DataFrame,
                                expected_dates):

        logger.debug("Creating buyer features ...")
        # go to buyer dataset and creates lagged features
        feat_df = pd.DataFrame(index=expected_dates)

        # Add features suggested from buyer to predict buyer_y target
        buyer_feature_list_ = [f"seller__{x}" for x in suggested_features]
        buyer_feat_ = market_features[buyer_feature_list_].copy()
        buyer_feat_.columns = [f"self__{x}" for x in suggested_features]
        feat_df = feat_df.join(buyer_feat_)

        if True:
            from .preprocessing.feature_engineering.autocorrelation import autocorrelation_analysis  # noqa
            from .preprocessing.feature_engineering.construct_inputs_funcs import construct_inputs_from_lags  # noqa
            target_col = "target"
            # Filter data for specific seller resource:
            _dataset = buyer_y[[target_col]]

            # Compute auto-correlation analysis (ACF)
            ac_analysis = autocorrelation_analysis(
                dataset=_dataset,
                acf_kwargs=settings.AutocorrelationAnalysis.acf_kwargs,
                target_col=target_col,
                forecast_horizon=self.FORECAST_HORIZON,
            )

            # Find hourly lags (based on ac analysis) & add to predictors
            hourly_lags = [-x[0] for x in ac_analysis["general"]["acf"]]

            # Find backup lag reference (difference between last forecast date
            # and last date available in the dataset)
            _last_date = _dataset.dropna().index[-1]
            lag_ref = -int((self.forecast_range[-1] - _last_date).total_seconds() / 3600)  # noqa
            hourly_lags.extend([lag_ref])
            # Remove duplicate lags (e.g., added by backup lag) but keep order:
            unique_lags = list(dict.fromkeys(hourly_lags))
            # Find hourly lags (based on ac analysis) & add to predictors
            lag_vars = {target_col: [('hour', list(unique_lags))]}
            lag_name_prefix = f"self__{target_resource_id}__"
            feat_df = construct_inputs_from_lags(
                df=_dataset,
                inputs=feat_df,
                lag_vars=lag_vars,
                lag_tz=self.MARKET_TZ,
                infer_dst_lags=False,
                lag_name_prefix=lag_name_prefix
            )

            # Check, for the forecast range, which of the ACF lags are valid
            check_nulls = feat_df.loc[self.forecast_range].isnull().any()  # noqa
            # get name (index) of columns with information available
            valid_lags = check_nulls[check_nulls == False].index
            if len(valid_lags) == 0:
                feat_df = feat_df[[]]  # empty dataframe
            else:
                feat_df = feat_df[valid_lags]  # select valid lags
                feat_df = feat_df.dropna()

        logger.debug("Creating buyer features ... Ok!")
        return feat_df

    def __process_features(self, market_x, buyer_x, buyer_y):
        launch_time_ = self.launch_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        # Join market and buyer features:
        features_ = market_x.join(buyer_x, how="right")
        # Prepare train dataset:
        train_features = features_[:launch_time_].join(buyer_y).dropna(subset=["target"])  # noqa
        # Remove "target" variable from train dataset:
        train_targets = train_features.pop("target").to_frame()
        # Test features (variables available for all dates since launch time)
        test_features = features_.loc[self.forecast_range]
        # Fill NaN - todo: find better imputation strategy (or drop NaN train)
        train_features.fillna(train_features.mean(), inplace=True)
        return train_features, train_targets, test_features

    def calculate_payment_and_forecast(self,
                                       buyer_cls,
                                       market_x_full: pd.DataFrame):

        # -- Load Buyer data
        logger.info(f"Processing buyer {buyer_cls.resource_id} bid ...")
        resource_id = buyer_cls.resource_id
        user_id = buyer_cls.user_id
        bid_price = buyer_cls.initial_bid
        bid_id = buyer_cls.market_bid_id
        max_payment = buyer_cls.max_payment
        buyer_y = buyer_cls.y[["value"]].copy()
        buyer_y.rename(columns={"value": "target"}, inplace=True)
        # Features suggested by buyer to predict this resource:
        suggested_features = buyer_cls.features_list
        # Buyer Features specified by agent to use in his model:
        user_features_list = self.users_data[user_id].user_features_list
        # Gain function:
        gain_func = buyer_cls.gain_func
        logger.debug(f"\nResource ID: {resource_id}"
                     f"\nUser ID:{user_id}"
                     f"\nBid Price:{bid_price}"
                     f"\nMax.Payment:{max_payment}"
                     f"\nGain Function:{gain_func}"
                     f"\nlen(y):{len(buyer_y)}"
                     )

        # -- Failure scenario return:
        fail_return = {"market_fee": 0, "payment": 0, "gain_func": gain_func,
                       "gain": 0, "final_bid": bid_price, "user_id": user_id,
                       "resource_id": resource_id, "forecasts": None}

        # -- Feature Engineering (own data) & select market features
        # Pre-process buyer data
        buyer_y = self.__preprocess_buyer_data(
            data=buyer_y,
            expected_dates=market_x_full.index,
        )
        # Buyer features:
        buyer_x = self.__create_buyer_features(
            buyer_y=buyer_y,
            target_resource_id=resource_id,
            suggested_features=suggested_features,
            market_features=market_x_full,
            expected_dates=market_x_full.index,
        )
        # Check if buyer feature set is empty. If so, cancels process:
        if buyer_x.empty:
            logger.warning(f"Buyer {user_id} resource {resource_id} "
                           f"features dataset is empty. Aborting forecast.")

            return fail_return

        # Select market features (all agents but buyer_id)
        logger.debug("Selecting market features ...")
        # -- Remove features from this user not suggested for this forecast
        market_x = self.__select_market_features(
            resource_id=resource_id,
            user_features_list=user_features_list,
            market_x_full=market_x_full
        )

        # -- Feature selection preprocessing
        if self.auto_feature_selection:
            market_x.index.name, buyer_x.index.name, buyer_y.index.name = 'datetime', 'datetime', 'datetime'  # noqa
            fs_params = settings.FeaturePreprocess.feature_selection
            feature_engineering = FeatureProcess(seed=fs_params['seed'],
                                                 method_name=fs_params['method_fs'],
                                                 type_selection=fs_params['type_selection'],
                                                 percentile=fs_params['percentile'],
                                                 threshold=fs_params['threshold'],
                                                 significance_level=fs_params['significance_level'],
                                                 nr_neighbors=fs_params['nr_neighbors'],
                                                 path_to_save_fs=fs_params['path_to_save_fs'],
                                                 dir_fs=fs_params['dir_fs'],
                                                 filename_scores=fs_params['filename_scores'],
                                                 filename_fs=fs_params['filename_fs'],
                                                 file_format=fs_params['format'])
            dict_results = feature_engineering.feature_selection(
                dfx_seller=market_x,
                dfx_buyer=buyer_x,
                dfy_buyer=buyer_y,
                save=True
            )
            feat_list, nr_feat_sel = feature_engineering.get_feature_selected(
                dict_results=dict_results)
            market_x = feature_engineering.get_feature_selection_df(
                dfx_seller=market_x,
                list_feature_selected=feat_list,
                nr_feature_selected=nr_feat_sel)

        # -- Features & targets arrays:
        train_features, train_targets, test_features = self.__process_features(
            market_x=market_x,
            buyer_x=buyer_x,
            buyer_y=buyer_y,
        )

        # Check if there are sufficient input samples to perform a forecast
        # for the test set:
        if test_features.isnull().any().any():
            logger.error(f"Error! User {user_id} resource {resource_id} "
                         f"has NaNs on the forecast "
                         f"inputs (test set). Aborting forecast.")
            return fail_return

        # Get features names and indexes
        sellers_features_name = list(market_x.columns)
        buyer_features_name = list(buyer_x.columns)
        train_features_name = list(train_features.columns)
        sellers_features_idx = [train_features_name.index(x) for x in sellers_features_name]
        buyer_features_idx = [train_features_name.index(x) for x in buyer_features_name]
        # -- Convert train data to numpy arrays (speed up)
        train_features = train_features.values
        train_targets = train_targets.values
        # features = market_x.join(buyer_x).values
        logger.debug("Selecting market features ... Ok!")

        # -- Buyer Payment Calculation
        run_cycle = True  # Repeat while buyer_payment > max_payment
        logger.debug(f"Calculating payment for resource ID {resource_id} ...")
        t0 = time()
        while run_cycle:
            # Calculate buyer payment:
            noisy_train_features, gain, payment = calc_buyer_payment(
                features=train_features,
                targets=train_targets,
                bid_price=bid_price,
                gain_func=gain_func,
                market_price=self.mkt_sess.market_price,
                b_min=self.mkt_sess.b_min,
                b_max=self.mkt_sess.b_max,
                epsilon=self.mkt_sess.epsilon,
                n_hours=self.N_HOURS,
                buyer_features_idx=buyer_features_idx,
                sellers_features_idx=sellers_features_idx
            )
            if payment <= max_payment:
                # Finish process if payment <= max_payment
                market_fee = payment * self.MARKET_FEE_PCT
                run_cycle = False
                logger.debug(f"\nGain: {gain}"
                             f"\nPayment: {payment}")
            else:
                # Else, affect buyer bid (bid-epsilon) & repeat:
                logger.warning(f"Payment ({payment}) higher than "
                               f"max_payment ({max_payment})!")  # noqa
                bid_price = max(0, bid_price - self.mkt_sess.epsilon)
                logger.warning(f"Bid price readjusted to {bid_price}. "
                               f"Recomputing ...")

        logger.debug(f"Calculating payment for resource ID {resource_id} ... "
                     f"Ok! ({time() - t0:.2f}s)")

        # -- Create Forecasts
        logger.debug("Creating forecasts ...")
        forecasts = create_forecast(
            train_features=train_features,
            train_targets=train_targets,
            test_features_df=test_features,
        )

        if self.db_uploads:
            inserted = upload_forecasts(
                market_session_id=self.mkt_sess.session_id,
                request=self.launch_time,
                user_id=user_id,
                resource_id=resource_id,
                forecasts=forecasts,
                table_name=self.FORECASTS_TABLE
            )
            if inserted:
                update_bid_has_forecast(
                    user_id=user_id,
                    bid_id=bid_id,
                    table_name=self.BIDS_TABLE
                )

        logger.info(f"Processing buyer {buyer_cls.resource_id} bid ... Ok!")
        return {
            "features": train_features,
            "noisy_train_features": noisy_train_features,
            "train_features_name": train_features_name,
            "market_fee": market_fee,
            "payment": payment,
            "targets": train_targets,
            "gain_func": gain_func,
            "gain": gain,
            "final_bid": bid_price,
            "initial_bid": buyer_cls.initial_bid,
            "resource_id": resource_id,
            "user_id": user_id,
            "buyer_features_name": list(buyer_x.columns),
            "sellers_features_name": sellers_features_name,
            "buyer_features_idx": buyer_features_idx,
            "sellers_features_idx": sellers_features_idx,
            "forecasts": forecasts
        }

    def define_sellers_revenue(self):
        for i, input_kwargs in enumerate(self.buyer_outputs):
            if input_kwargs["payment"] > 0:
                t0 = time()
                logger.debug("Distributing revenue ...")
                # Distribute payment by sellers:
                sellers_id_list = list(self.sellers_data.keys())
                buyer_resource_id = input_kwargs["resource_id"]
                user_features_list = self.users_data[input_kwargs["user_id"]].user_features_list  # noqa
                sellers_revenue_split = calc_sellers_revenue(
                    buyer_resource_id=buyer_resource_id,
                    user_features_list=user_features_list,
                    noisy_features=input_kwargs["noisy_train_features"],
                    targets=input_kwargs["targets"],
                    gain_func=input_kwargs["gain_func"],
                    buyer_resource_payment=input_kwargs["payment"],
                    buyer_market_fee=input_kwargs["market_fee"],
                    sellers_id_list=sellers_id_list,
                    sellers_features_name=input_kwargs["sellers_features_name"],  # noqa
                    buyer_features_idx=input_kwargs["buyer_features_idx"],
                    K=self.REVENUE_K,
                    lambd=self.REVENUE_LAMBDA,
                    n_hours=self.N_HOURS,
                )
                logger.debug(
                    f"Sellers revenue split:\n{sellers_revenue_split}")
                # Assign seller revenue
                logger.debug("Storing revenue in sellers class ...")
                self.buyers_data[buyer_resource_id].set_payment_split(
                    sellers_revenue_split
                )
                for seller_id in sellers_revenue_split.keys():
                    _r = sellers_revenue_split[seller_id].get("abs_revenue", 0)
                    _sv = sellers_revenue_split[seller_id].get("shapley_value", 0)
                    self.sellers_data[seller_id].increment_revenue(_r)
                    self.sellers_data[seller_id].increment_shapley_value(_sv)
                logger.debug("Storing revenue in sellers class ... Ok!")
                logger.debug(f"Distributing revenue ... Ok! "
                             f"({time() - t0:.2f}s)")

    def save_session_results(self, save_forecasts=False):
        """
        Update buyer's & seller's Classes w/ session results

        """
        logger.info("Saving session results ...")
        for cls in self.buyers_data.values():
            self.mkt_sess.set_buyer_result(cls)
            if save_forecasts:
                self.mkt_sess.set_buyer_forecasts(cls)

        for cls in self.sellers_data.values():
            self.mkt_sess.set_seller_result(cls)
        logger.info("Saving session results ... Ok!")

    def validate_session_results(self, raise_exception=True):
        # Confirm if there are no errors in market session results:
        fee = self.mkt_sess.total_market_fee
        deposits = [v["max_payment"] for k, v in self.mkt_sess.buyers_results.items()]
        payments = [v["has_to_pay"] for k, v in self.mkt_sess.buyers_results.items()]
        revenues = [v["has_to_receive"] for k, v in self.mkt_sess.sellers_results.items()]
        sv = [v["shapley_value"] for k, v in self.mkt_sess.sellers_results.items()]
        logger.info("")
        logger.info("Validating session results:")
        logger.debug("Market fee:", self.mkt_sess.total_market_fee)
        logger.debug(f"Buyers deposits: {deposits} // Total: {sum(deposits)}")
        logger.debug(f"Buyers payments: {payments} // Total: {sum(payments)}")
        logger.debug(f"Sellers revenue: {revenues} // Total: {sum(revenues)}")
        logger.debug(f"Sellers shapley values: {sv} // Total: {sum(sv)}")
        logger.debug(f"Market fee: {fee}")
        result = sum(payments) - fee - sum(revenues)
        logger.debug(f"""
        Validation (1) - Buyer payments should be distributed by market (fees) and sellers revenues
        Payments({sum(payments)}) - Market({fee}) - Revenues({sum(revenues)}) = Zero({result})
        """)
        is_valid = round(result, 9) == 0.0
        logger.info(f"Valid Session: {is_valid}")
        if (not is_valid) and raise_exception:
            raise ValueError("Payments - Fee - Revenues != 0. Invalid session.")

    def payment_and_revenue_per_user(self):
        for resource_data in self.buyers_data.values():
            user_id = resource_data.user_id
            has_to_pay = resource_data.has_to_pay
            self.users_data[user_id].sum_payment(has_to_pay)

        for resource_data in self.sellers_data.values():
            user_id = resource_data.user_id
            has_to_receive = resource_data.has_to_receive
            self.users_data[user_id].sum_revenue(has_to_receive)

    def set_forecast_range(self):
        self.forecast_start = self.launch_time.replace(minute=0, second=0, microsecond=0)
        self.forecast_end = self.forecast_start + pd.DateOffset(hours=self.FORECAST_HORIZON)  # noqa
        self.forecast_range = pd.date_range(start=self.forecast_start,
                                            end=self.forecast_end,
                                            freq='h',
                                            tz=self.MARKET_TZ,
                                            inclusive="right")

    def define_payments_and_forecasts(self):
        """
        Run current market session

        Steps:
            1. Create market dataset (aggregate sellers measurements data)
            2. Create market features
            3. Process bids & forecasts for each buyer agent. For each buyer:
                3.1. Load Buyer data
                3.2. Feature Engineering (own data) & select market features
                3.3. Buyer Payment Calculation
                3.4. Create Forecasts
            4. Sellers Revenue Calculation
            5. Save session results

        """
        logger.info("-" * 70)
        logger.info(f"Running session {self.mkt_sess.session_id}...")
        if len(self.buyers_data) == 0:
            e_msg = "Error! Insufficient buyers bids to start a new session."
            logger.error(e_msg)
            raise NoMarketBuyersExceptions(e_msg)

        if len(self.users_data) == 0:
            e_msg = "Error! Users data not loaded. Use self.users_data()."
            logger.error(e_msg)
            raise NoMarketUsersExceptions(e_msg)

        # -- 0. Define forecast range based on launch time & horizon:
        self.set_forecast_range()

        # -- 1. Create market dataset (aggregate sellers measurements data)
        market_df = self.__create_market_dataset()

        # -- 2. Create market features
        market_x_full = self.__create_market_features(market_df=market_df)

        # -- 3. Process payment & forecasts for each buyer resource
        self.buyer_outputs = Parallel(n_jobs=self.n_jobs)(
            delayed(
                self.calculate_payment_and_forecast
            )(buyer_cls, market_x_full)
            for buyer_cls in self.buyers_data.values()
        )

        # -- 3.1 Store results in each buyer cls & sum market fees:
        for out in self.buyer_outputs:
            self.buyers_data[out["resource_id"]].set_payment(out["payment"])
            self.buyers_data[out["resource_id"]].set_gain(out["gain"])
            self.buyers_data[out["resource_id"]].set_final_bid(out["final_bid"])  # noqa
            self.buyers_data[out["resource_id"]].set_forecasts(out["forecasts"])  # noqa
            self.mkt_sess.add_market_fee(
                resource_id=out["resource_id"],
                value=out["market_fee"]
            )

    def update_market_price(self):
        logger.info("-" * 70)
        logger.info("Updating market prices for next session ...")
        probs = []
        price_weights = self.mkt_sess.prev_weights_p
        # -- Iterate through each buyer inputs & calc price weights
        for i, input_kwargs in enumerate(self.buyer_outputs):
            logger.debug(f"Iteration #{i + 1}")
            probs, price_weights = market_price_update_parallel(
                w=price_weights,
                Bmin=self.mkt_sess.b_min,
                Bmax=self.mkt_sess.b_max,
                epsilon=self.mkt_sess.epsilon,
                delta=self.mkt_sess.delta,
                n_hours=self.N_HOURS,
                possible_p=self.mkt_sess.possible_p,
                features=input_kwargs["features"],
                targets=input_kwargs["targets"],
                gain_func=input_kwargs["gain_func"],
                bid_price=input_kwargs["initial_bid"],
                buyer_features_idx=input_kwargs["buyer_features_idx"],
                sellers_features_idx=input_kwargs["sellers_features_idx"],
                n_jobs=self.n_jobs
            )
            logger.debug(f"Current price weights: {price_weights}")
            logger.debug(f"Current probs: {probs}")

        # -- Define & store market price & weights for next session:
        # Calculate market price for next session:
        next_market_price = sum(probs * self.mkt_sess.possible_p)
        next_market_price = (next_market_price // self.mkt_sess.epsilon + 1)
        next_market_price *= self.mkt_sess.epsilon
        logger.debug(f"Next market price: {next_market_price}")
        # Save next market price & weights:
        self.mkt_sess.set_next_market_price(next_market_price)
        self.mkt_sess.set_next_price_weights(price_weights)
        logger.info("Updating market prices for next session ... Ok!")

    def process_payments(self, api_controller=None):
        if api_controller is None:
            raise AttributeError("Error! Must provide an api controller "
                                 "to process payments.")

        # -- Market Session ID:
        market_session_id = self.mkt_sess.session_id
        # -- Process market fee payment (to market superuser):
        fees_iota = convert_mi_to_i(self.mkt_sess.total_market_fee)
        # -- Todo: Adicionar Controlo de exceptions:
        api_controller.post_session_market_fee(
            session_id=market_session_id,
            fee_amount=fees_iota,
        )
        # -- Process payments for agents (updated market account)
        for resource_id, buyer_info in self.mkt_sess.buyers_results.items():
            payment_iota = convert_mi_to_i(buyer_info["has_to_pay"])
            user_id = buyer_info["user_id"]
            api_controller.post_session_balance(
                user_id=user_id,
                resource_id=resource_id,
                session_id=market_session_id,
                amount=-payment_iota,
                transaction_type="payment"
            )
        # -- Process revenue for agents (updated market account)
        for resource_id, seller_info in self.mkt_sess.sellers_results.items():
            revenue_iota = convert_mi_to_i(seller_info["has_to_receive"])
            user_id = seller_info["user_id"]
            api_controller.post_session_balance(
                user_id=user_id,
                resource_id=resource_id,
                session_id=market_session_id,
                amount=revenue_iota,
                transaction_type="revenue"
            )

    def open_next_session(self, api_controller=None):
        if api_controller is None:
            raise AttributeError("Error! Must provide an api controller "
                                 "to process payments.")

        # Conversion from MIOTA to IOTA:
        market_price_ = convert_mi_to_i(self.mkt_sess.next_market_price)
        b_min_ = convert_mi_to_i(self.mkt_sess.b_min)
        b_max_ = convert_mi_to_i(self.mkt_sess.b_max)

        # -- Todo: Adicionar Controlo de exceptions:
        # -- Todo: verificar se à 6ta sessão já n deviamos mudar session_date
        api_controller.create_market_session(
            session_number=self.mkt_sess.session_number + 1,
            market_price=market_price_,
            b_min=b_min_,
            b_max=b_max_,
            n_price_steps=self.mkt_sess.n_price_steps,
            delta=self.mkt_sess.delta
        )
        api_controller.post_session_weights(
            session_id=self.mkt_sess.session_id + 1,
            weights_p=self.mkt_sess.next_weights_p
        )
