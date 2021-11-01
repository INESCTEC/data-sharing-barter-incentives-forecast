import pandas as pd
import datetime as dt

from time import time
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

# -- Mock data imports:
from src.market.helpers.model_helpers import create_forecast
from src.market.helpers.units_helpers import convert_mi_to_i


class MarketClass:
    N_HOURS = 24 * 31                  # no. hours in evaluation period
    FORECAST_HORIZON = 1               # forecast horizon in market
    N_HOURS_IN_HIST = 8760             # no. hours in historical data
    MARKET_FEE_PCT = 0.05              # market fee applied to buyer payment
    REVENUE_K = 5
    REVENUE_LAMBDA = 1
    FORECASTS_TABLE = "market_forecasts"
    MEASUREMENTS_TABLE = "market_forecasts"
    BIDS_TABLE = "market_session_bid"

    def __init__(self, n_jobs=-1):
        self.users_data = {}
        self.users_list = []
        self.users_resources_list = []
        self.buyers_data = {}
        self.sellers_data = {}
        self.mkt_sess = None
        self.launch_time = None
        self.finish_time = None
        self.buyer_outputs = []
        self.n_jobs = n_jobs

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
        logger.info(f"Session ID: {self.mkt_sess.session_id}")
        logger.info(f"Buyers:\n"
                    f"{json.dumps(self.mkt_sess.buyers_results, indent=2)}")
        logger.info("-")
        logger.info(f"Sellers:\n"
                    f"{json.dumps(self.mkt_sess.sellers_results, indent=2)}")
        logger.info("-")
        logger.info(f"Market Session:"
                    f"\n{json.dumps(self.mkt_sess.details, indent=2)}")

    def load_resources_bids(self, bids: list):
        if (not isinstance(bids, list)) or \
                (len(bids) > 0) and (not isinstance(bids[0], dict)):
            raise TypeError("Error! bids argument must be a list of dicts")

        for buyer_bid in bids:
            # Init Buyer class with each bid information:
            cls = BuyerClass(
                user_id=buyer_bid["user"],
                resource_id=buyer_bid["resource"],
                initial_bid=buyer_bid["bid_price"],
                max_payment=buyer_bid["max_payment"],
                gain_func=buyer_bid["gain_func"],
                market_bid_id=buyer_bid["id"]
            ).validate_attributes()
            self.buyers_data[cls.resource_id] = cls

    def load_users_resources(self, users_resources: list):
        if not isinstance(users_resources, list):
            raise TypeError("Error! a list of resources must be provided")
        # Init Seller class with each seller identification:
        self.users_resources = users_resources
        for resource_data in self.users_resources:
            user_id = resource_data["user"]
            cls = SellerClass(
                user_id=user_id,
                resource_id=resource_data["id"],
            )
            cls.validate_attributes()
            self.sellers_data[cls.resource_id] = cls
            if user_id not in self.users_list:
                self.users_list.append(user_id)

    def load_resources_measurements(self, measurements: dict):
        if not isinstance(measurements, dict):
            raise TypeError("Error! measurements arg. must be a dict")
        # Intersection - agents that are sellers & buyers
        resource_list = set(list(self.buyers_data.keys()) +
                            list(self.sellers_data.keys()))
        # Assign measurements data to each agent class:
        default_df = pd.DataFrame(columns=["datetime", "value"])
        for resource_id in sorted(resource_list):
            # Fetch agent data (empty dataset if key not found)
            _df = measurements.get(resource_id, default_df)
            if resource_id in self.buyers_data:
                self.buyers_data[resource_id].set_measurements(_df)
            if resource_id in self.sellers_data:
                self.sellers_data[resource_id].set_measurements(_df)

    def load_users(self):
        for user_id in self.users_list:
            self.users_data[user_id] = UserClass(user_id=user_id)

    @staticmethod
    def __preprocess_buyer_data(data, expected_dates):
        """
        Resample data to hourly resolution
        Reindex so missing dates are market as NA

        :param data:
        :param expected_dates:
        :return:
        """
        data = data.resample('H').mean()
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
        # 1. Define expected datetime range:
        logger.info("Creating market dataset ...")
        _end_date = self.launch_time.replace(minute=0, second=0, microsecond=0)  # noqa
        _end_date = _end_date + pd.DateOffset(hours=self.FORECAST_HORIZON)
        _lookback_time = self.N_HOURS_IN_HIST - 1 + self.FORECAST_HORIZON
        _range = pd.date_range(
            start=_end_date - pd.DateOffset(hours=_lookback_time),
            end=_end_date,
            tz="utc",
            freq="H"
        )
        # 2. Add sellers data:
        market_df = pd.DataFrame(index=_range)
        for seller_id, seller_cls in self.sellers_data.items():
            df_ = seller_cls.y[["value"]]
            df_ = df_.rename(columns={"value": seller_id})
            df_ = df_.resample("H").mean()
            market_df = market_df.join(df_, how="left")
        # Check if there is no market data:
        if market_df.dropna(how="all").empty:
            e_msg = "Error! No market dataset available. Terminating ..."
            logger.error(e_msg)
            raise NoMarketDataException(e_msg)
        else:
            logger.info("Creating market dataset ... Ok!")
            return market_df

    def __create_market_features(self, market_df: pd.DataFrame):
        logger.info("Creating market features ...")
        # go to market dataset and ignore agent_id measurements
        # todo: create lagged features (selection based on CCF)
        feat_df = pd.DataFrame(index=market_df.index)  # assure idx = buyer
        for seller_id in market_df.columns:
            for i in range(0, 1):
                _name = f"seller__{seller_id}__l{i}"
                feat_df.loc[:, _name] = market_df[seller_id].shift(self.FORECAST_HORIZON + i)  # noqa
        # todo: verificar que features n existem no horizonte de previsão
        #  e só depois substituir NaN
        feat_df.dropna(how="all", inplace=True)
        feat_df.fillna(0, inplace=True)
        logger.info("Creating market features ... Ok!")
        return feat_df

    @staticmethod
    def __select_market_features(resource_id: str,
                                 market_x_full: pd.DataFrame):
        """
        Select all market features except the ones for a specif agent_id

        :param resource_id: Agent identifier
        :param market_x_full: Market dataset
        :return:
        """
        # todo: -- market_x_full -> adaptar para criar vários lags diferentes (por seller)
        #  depois, filtrar sellers q n têm dados para as ultimas X horas (X = nº lags)
        #  depois, selecionar features com maior correlação com série de buyer
        _cols = [x for x in market_x_full.columns
                 if int(x.split('__')[1]) != resource_id]
        return market_x_full[_cols]

    def __create_buyer_features(self,
                                buyer_y: pd.DataFrame,
                                expected_dates):
        logger.debug("Creating buyer features ...")
        # go to buyer dataset and creates lagged features
        # todo: (selection based on ACP / PACF)
        feat_df = pd.DataFrame(index=expected_dates)
        for i in range(1, 2):
            feat_df.loc[:, f"self__l{i}"] = buyer_y["target"].shift(self.FORECAST_HORIZON)  # noqa
        feat_df.fillna(0, inplace=True)
        logger.debug("Creating buyer features ... Ok!")
        return feat_df

    def __process_features(self, market_x, buyer_x, buyer_y):
        launch_time_ = self.launch_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        # Join market and buyer features:
        features_ = market_x.join(buyer_x)
        # Prepare train dataset:
        train_features = features_[:launch_time_].join(buyer_y).dropna(subset=["target"])  # noqa
        # Remove "target" variable from train dataset:
        train_targets = train_features.pop("target").to_frame()
        # Test features (variables available for all dates since launch time)
        test_features = features_[launch_time_:]
        return train_features, train_targets, test_features

    def payment_and_forecast(self,
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
        gain_func = buyer_cls.gain_func
        logger.debug(f"\nResource ID: {resource_id}"
                     f"\nUser ID:{user_id}"
                     f"\nBid Price:{bid_price}"
                     f"\nMax.Payment:{max_payment}"
                     f"\nGain Function:{gain_func}"
                     f"\nlen(y):{len(buyer_y)}"
                     )
        # -- Feature Engineering (own data) & select market features
        # Pre-process buyer data
        buyer_y = self.__preprocess_buyer_data(
            data=buyer_y,
            expected_dates=market_x_full.index,
        )
        # Buyer features:
        buyer_x = self.__create_buyer_features(
            buyer_y=buyer_y,
            expected_dates=market_x_full.index,
        )
        # Select market features (all agents but buyer_id)
        logger.debug("Selecting market features ...")
        market_x = self.__select_market_features(
            resource_id=resource_id,
            market_x_full=market_x_full
        )
        # -- Features & targets arrays:
        sellers_features_name = list(market_x.columns)
        train_features, train_targets, test_features = self.__process_features(
            market_x=market_x,
            buyer_x=buyer_x,
            buyer_y=buyer_y,
        )
        # todo: avaliar test_features e decidir que sellers têm direito a
        #  participar no mercado para este buyer remover sellers com + do
        #  que 3 NaN -> com menos do que 3, interpolar
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
            "market_fee": market_fee,
            "payment": payment,
            "targets": train_targets,
            "gain_func": gain_func,
            "gain": gain,
            "final_bid": bid_price,
            "initial_bid": buyer_cls.initial_bid,
            "resource_id": resource_id,
            "user_id": user_id,
            "sellers_features_name": sellers_features_name,
        }

    def sellers_revenue(self):
        for i, input_kwargs in enumerate(self.buyer_outputs):
            if input_kwargs["payment"] > 0:
                t0 = time()
                logger.debug("Distributing revenue ...")
                # Distribute payment by sellers:
                sellers_id_list = list(self.sellers_data.keys())
                buyer_resource_id = input_kwargs["resource_id"]
                sellers_revenue_split = calc_sellers_revenue(
                    buyer_resource_id=buyer_resource_id,
                    noisy_features=input_kwargs["noisy_train_features"],
                    targets=input_kwargs["targets"],
                    gain_func=input_kwargs["gain_func"],
                    buyer_resource_payment=input_kwargs["payment"],
                    buyer_market_fee=input_kwargs["market_fee"],
                    sellers_id_list=sellers_id_list,
                    sellers_features_name=input_kwargs["sellers_features_name"],  # noqa
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
                    self.sellers_data[seller_id].increment_revenue(_r)
                logger.debug("Storing revenue in sellers class ... Ok!")
                logger.debug(f"Distributing revenue ... Ok! "
                             f"({time() - t0:.2f}s)")

    def save_session_results(self):
        """
        Update buyer's & seller's Classes w/ session results

        """
        logger.info("Saving session results ...")
        for cls in self.buyers_data.values():
            self.mkt_sess.set_buyer_result(cls)
        for cls in self.sellers_data.values():
            self.mkt_sess.set_seller_result(cls)
        logger.info("Saving session results ... Ok!")

        # Confirm if there are no errors in market session results:
        fee = self.mkt_sess.total_market_fee
        payments = [v["has_to_pay"] for k, v in self.mkt_sess.buyers_results.items()]
        revenues = [v["has_to_receive"] for k, v in self.mkt_sess.sellers_results.items()]
        logger.info("")
        logger.info("Validating session results:")
        logger.debug("Market fee:", self.mkt_sess.total_market_fee)
        logger.debug("Buyers payments:")
        logger.debug(payments)
        logger.debug("Total Buyers payments:", sum(payments))
        logger.debug("Sellers revenue:")
        logger.debug(revenues)
        logger.debug("Total Sellers Revenue:", sum(revenues))
        result = sum(payments) - fee - sum(revenues)
        logger.debug(f"Validation: {sum(payments)} - {fee} - {sum(revenues)} = {result}")
        is_valid = round(result, 9) == 0.0
        logger.info(f"Valid Session: {is_valid}")
        if not is_valid:
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

    def run_session(self):
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

        # -- 1. Create market dataset (aggregate sellers measurements data)
        market_df = self.__create_market_dataset()
        # -- 2. Create market features (NaNs filled with Zeros)
        market_x_full = self.__create_market_features(market_df=market_df)
        # -- 3. Process payment & forecasts for each buyer resource
        self.buyer_outputs = Parallel(n_jobs=self.n_jobs)(
            delayed(self.payment_and_forecast)(buyer_cls, market_x_full)
            for buyer_cls in self.buyers_data.values()
        )

        # -- 3.1 Store results in each buyer cls & sum market fees:
        for out in self.buyer_outputs:
            self.buyers_data[out["resource_id"]].set_payment(out["payment"])
            self.buyers_data[out["resource_id"]].set_gain(out["gain"])
            self.buyers_data[out["resource_id"]].set_final_bid(out["final_bid"])  # noqa
            self.mkt_sess.add_market_fee(
                resource_id=out["resource_id"],
                value=out["market_fee"]
            )

        # -- 4. Calculate Revenue per Seller Resource
        self.sellers_revenue()

        # -- 5. Sum Payment / Revenue per User:
        self.payment_and_revenue_per_user()

        # -- 6. Save session results
        self.save_session_results()

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
            market_session=market_session_id,
            amount=fees_iota,
        )
        # -- Process payments for agents (updated market account)
        for buyer_id, buyer_info in self.mkt_sess.buyers_results.items():
            payment_iota = convert_mi_to_i(buyer_info["has_to_pay"])
            api_controller.post_session_balance(
                user=buyer_id,
                market_session=market_session_id,
                amount=-payment_iota,
                transaction_type="payment"
            )
        # -- Process revenue for agents (updated market account)
        for seller_id, seller_info in self.mkt_sess.sellers_results.items():
            revenue_iota = convert_mi_to_i(seller_info["has_to_receive"])
            api_controller.post_session_balance(
                user=seller_id,
                market_session=market_session_id,
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
