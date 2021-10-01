import pandas as pd

from time import time
from loguru import logger
from joblib import Parallel, delayed

# -- Helper funcs:
from core.market.helpers.market_helpers import (
    calc_buyer_payment,
    calc_sellers_revenue,
    market_price_update_parallel,
)

# -- Market entities classes:
from core.market.BuyerClass import BuyerClass
from core.market.SellerClass import SellerClass
from core.market.SessionClass import SessionClass
from core.market.util.custom_exceptions import (
    NoMarketDataException,
    NoMarketBuyersExceptions
)

# -- Mock data imports:
from core.market.helpers.model_helpers import create_forecast
from core.market.helpers.units_helpers import convert_mi_to_i


class MarketClass:
    N_HOURS = 24 * 31                  # no. hours in evaluation period
    FORECAST_HORIZON = 1               # forecast horizon in market
    N_HOURS_IN_HIST = 8760             # no. hours in historical data
    MARKET_FEE_PCT = 0.05              # market fee applied to buyer payment
    REVENUE_K = 5
    REVENUE_LAMBDA = 1

    def __init__(self, n_jobs=-1):
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
            session_id=session_data["market_session_id"],
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

    def load_buyers_bids(self, bids: list):
        if (not isinstance(bids, list)) or \
                (len(bids) > 0) and (not isinstance(bids[0], dict)):
            raise TypeError("Error! bids argument must be a list of dicts")
        # Init Buyer class with each bid information:
        for buyer_bid in bids:
            # todo: check if bid is confirmed before accepting
            cls = BuyerClass(
                identifier=buyer_bid["user"],
                initial_bid=buyer_bid["bid_price"],
                max_payment=buyer_bid["max_payment"],
                gain_func=buyer_bid["gain_func"]
            )
            cls.validate_attributes()
            self.buyers_data[cls.identifier] = cls

    def load_sellers(self, identifiers: list):
        if not isinstance(identifiers, list):
            raise TypeError("Error! a list of identifiers must be provided")
        # Init Seller class with each seller identification:
        for seller_id in identifiers:
            cls = SellerClass(identifier=seller_id)
            cls.validate_attributes()
            self.sellers_data[cls.identifier] = cls

    def load_agents_measurements(self, measurements: dict):
        if not isinstance(measurements, dict):
            raise TypeError("Error! measurements arg. must be a dict")
        # Intersection - agents that are sellers & buyers
        agent_list = set(list(self.buyers_data.keys()) +
                         list(self.sellers_data.keys()))
        for agent in sorted(agent_list):
            # Fetch agent data (empty dataset if key not found)
            _df = measurements.get(agent, pd.DataFrame())
            if agent in self.buyers_data:
                self.buyers_data[agent].set_measurements(_df)
            if agent in self.sellers_data:
                self.sellers_data[agent].set_measurements(_df)

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
        market_df = pd.DataFrame(index=_range)
        for seller_id, seller_cls in self.sellers_data.items():
            df_ = seller_cls.y.set_index("datetime")[["value"]]
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
            # Fill NaN w/ zeros & return market df
            return market_df.fillna(0)

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
        feat_df.fillna(0, inplace=True)
        logger.info("Creating market features ... Ok!")
        return feat_df

    @staticmethod
    def __select_market_features(agent_id: str,
                                 market_x_full: pd.DataFrame):
        """
        Select all market features except the ones for a specif agent_id

        :param agent_id: Agent identifier
        :param market_x_full: Market dataset
        :return:
        """
        _cols = [x for x in market_x_full.columns
                 if int(x.split('__')[1]) != agent_id]
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
        # Join market and buyer features:
        features_ = market_x.join(buyer_x)
        # Prepare train dataset:
        train_features = features_[:self.launch_time].join(buyer_y).dropna(subset=["target"])  # noqa
        # Remove "target" variable from train dataset:
        train_targets = train_features.pop("target").to_frame()
        # Test features (variables available for all dates since launch time)
        test_features = features_[self.launch_time:]
        return train_features, train_targets, test_features

    def payment_and_forecast(self,
                             buyer_cls,
                             market_x_full,
                             market_price,
                             b_min,
                             b_max,
                             epsilon,
                             n_hours
                             ):
        # -- Load Buyer data
        logger.info(f"Processing buyer {buyer_cls.identifier} bid ...")
        buyer_id = buyer_cls.identifier
        buyer_bid = buyer_cls.initial_bid
        max_pay_ = buyer_cls.max_payment
        buyer_y = buyer_cls.y.set_index("datetime")[["value"]]
        buyer_y.rename(columns={"value": "target"}, inplace=True)
        gain_func = buyer_cls.gain_func
        logger.debug(f"\n-- Buyer {buyer_id}"
                     f"\nBid:{buyer_bid}"
                     f"\nMax.Payment:{max_pay_}"
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
            agent_id=buyer_id,
            market_x_full=market_x_full
        )
        # -- Features & targets arrays:
        sellers_features_names = list(market_x.columns)
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
        logger.debug(f"Calculating buyer {buyer_id} payment ...")
        while run_cycle:
            # Calculate buyer payment:
            t0 = time()
            noisy_train_features, gain, payment = calc_buyer_payment(
                features=train_features,
                targets=train_targets,
                bid_price=buyer_bid,
                gain_func=gain_func,
                market_price=market_price,
                b_min=b_min,
                b_max=b_max,
                epsilon=epsilon,
                n_hours=n_hours,
            )
            if payment <= max_pay_:
                # Finish process if payment <= max_payment
                market_fee = payment * self.MARKET_FEE_PCT
                run_cycle = False
                logger.debug(f"\nGain: {gain}"
                             f"\nPayment: {payment}")
            else:
                # Else, affect buyer bid (bid-epsilon) & repeat:
                logger.warning(f"Payment ({payment}) higher than "
                               f"max_payment ({max_pay_})!")  # noqa
                buyer_bid = max(0, buyer_bid - self.mkt_sess.epsilon)
                logger.warning(f"Bid value readjusted to {buyer_bid}. "
                               f"Recomputing ...")

        logger.debug(f"Calculating buyer {buyer_id} payment ... Ok! "
                     f"({time() - t0:.2f}s)")  # noqa

        # -- Create Forecasts
        logger.debug("Creating forecasts ...")
        forecasts = create_forecast(
            buyer_id=buyer_id,
            train_features=train_features,
            train_targets=train_targets,
            test_features_df=test_features,
        )
        # todo: insert forecasts in BD:
        # todo: adicionar request aqui para avisar que user ja tem forecasts
        #  para sessao atual.
        self.upload_forecasts(user_id=buyer_id, forecasts=forecasts)
        logger.info(f"Processing buyer {buyer_id} bid ... Ok!")
        return {
            "features": train_features,
            "noisy_train_features": noisy_train_features,
            "buyer_market_fee": market_fee,
            "payment": payment,
            "targets": train_targets,
            "gain_func": gain_func,
            "gain": gain,
            "final_bid": buyer_bid,
            "initial_bid": buyer_cls.initial_bid,
            "buyer_id": buyer_id,
            "sellers_features_names": sellers_features_names,
        }

    def sellers_revenue(self):
        for i, input_kwargs in enumerate(self.buyer_outputs):
            if input_kwargs["payment"] > 0:
                t0 = time()
                logger.debug("Distributing revenue ...")
                # Distribute payment by sellers:
                sellers_id_list = list(self.sellers_data.keys())
                buyer_id = input_kwargs["buyer_id"]
                sellers_revenue_split = calc_sellers_revenue(
                    buyer_id=buyer_id,
                    noisy_features=input_kwargs["noisy_train_features"],
                    targets=input_kwargs["targets"],
                    gain_func=input_kwargs["gain_func"],
                    buyer_payment=input_kwargs["payment"],
                    buyer_market_fee=input_kwargs["buyer_market_fee"],
                    sellers_id_list=sellers_id_list,
                    sellers_features_names=input_kwargs["sellers_features_names"],  # noqa
                    K=self.REVENUE_K,
                    lambd=self.REVENUE_LAMBDA,
                    n_hours=self.N_HOURS,
                )
                logger.debug(
                    f"Sellers revenue split:\n{sellers_revenue_split}")  # noqa
                # Assign seller revenue
                logger.debug("Storing revenue in sellers class ...")
                self.buyers_data[buyer_id].set_payment_split(sellers_revenue_split)  # noqa
                for seller_id in sellers_revenue_split.keys():
                    _r = sellers_revenue_split[seller_id].get("abs_revenue", 0)
                    self.sellers_data[seller_id].increment_revenue(_r)
                logger.debug("Storing revenue in sellers class ... Ok!")
                logger.debug(
                    f"Distributing revenue ... Ok! ({time() - t0:.2f}s)")  # noqa

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
        # -- 1. Create market dataset (aggregate sellers measurements data)
        market_df = self.__create_market_dataset()
        # -- 2. Create market features
        market_x_full = self.__create_market_features(market_df=market_df)
        # -- 3. Process payment & forecasts for each buyer agent
        self.buyer_outputs = Parallel(
            n_jobs=self.n_jobs,
        )(delayed(self.payment_and_forecast)(
            buyer_cls,
            market_x_full,
            self.mkt_sess.market_price,
            self.mkt_sess.b_min,
            self.mkt_sess.b_max,
            self.mkt_sess.epsilon,
            self.N_HOURS
        ) for buyer_cls in self.buyers_data.values())
        # -- 3.1 Store results in each buyer cls & sum market fees:
        for out in self.buyer_outputs:
            self.buyers_data[out["buyer_id"]].set_payment(out["payment"])
            self.buyers_data[out["buyer_id"]].set_gain(out["gain"])
            self.buyers_data[out["buyer_id"]].set_final_bid(out["final_bid"])
            self.mkt_sess.add_market_fee(
                buyer_id=out["buyer_id"],
                value=out["buyer_market_fee"]
            )
        # -- 4. Calculate Sellers Revenue
        self.sellers_revenue()
        # -- 5. Save session results
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
        # -- Todo: Adicionar metodo para saber admin user ID:
        admin_user_id = 1
        api_controller.post_session_balance(
            user=admin_user_id,
            market_session=market_session_id,
            amount=fees_iota,
            transaction_type="revenue"
        )
        # -- Process payments for agents (updated market account)
        for buyer_id, buyer_info in self.mkt_sess.buyers_results.items():
            payment_iota = convert_mi_to_i(buyer_info["has_to_pay"])
            rsp = api_controller.post_session_balance(
                user=buyer_id,
                market_session=market_session_id,
                amount=-payment_iota,
                transaction_type="payment"
            )
            print()
        # -- Process revenue for agents (updated market account)
        for seller_id, seller_info in self.mkt_sess.sellers_results.items():
            revenue_iota = convert_mi_to_i(seller_info["has_to_receive"])
            rsp = api_controller.post_session_balance(
                user=seller_id,
                market_session=market_session_id,
                amount=revenue_iota,
                transaction_type="revenue"
            )
            print()

    def upload_forecasts(self, user_id, forecasts):
        # upload forecasts to DB
        # Run after payments are confirmed
        # todo: fazer upload de forecasts para cassandra
        # todo: atualizar campo "has_forecast" na bid para esta sessão
        pass

    def open_next_session(self, api_controller=None):
        if api_controller is None:
            raise AttributeError("Error! Must provide an api controller "
                                 "to process payments.")

        # Conversion from MIOTA to IOTA:
        market_price_ = convert_mi_to_i(self.mkt_sess.next_market_price)
        b_min_ = convert_mi_to_i(self.mkt_sess.b_min)
        b_max_ = convert_mi_to_i(self.mkt_sess.b_max)

        # p.e. verificar se à 6ta sessão já n deviamos mudar session_date
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
