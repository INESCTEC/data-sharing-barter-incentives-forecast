import numpy as np
import pandas as pd
import datetime as dt
from time import sleep
from loguru import logger

from conf import settings
from .api import Controller
from .wallet import WalletController, TangleController
from .market import MarketClass
from .market.helpers.api_helpers import (
    get_session_data,
    get_measurements_data,
    close_no_bids_session
)
from .market.helpers.units_helpers import (
    convert_session_data_to_mi,
    convert_buyers_bids_to_mi,
)

from .api.exception.APIException import *


class MarketController:
    def __init__(self):
        # Market Wallet Controller:
        self.wallet = WalletController()
        # Tangle Controller:
        self.tangle = TangleController()
        # Market API Controller:
        self.api = Controller()
        # todo: adicionar re-log caso token expire
        self.api.login(email=settings.MARKET_EMAIL,
                       password=settings.MARKET_PASSWORD)

    def deploy_market(self):
        # Create market roles:
        logger.debug("Creating market roles ...")
        self.api.create_market_role(role="buyer")
        self.api.create_market_role(role="seller")
        logger.debug("Creating market roles ... Ok!")

        # Create market wallet:
        logger.debug("Creating market wallet ...")
        self.wallet.create_wallet(store_mnemonic=True)
        self.wallet.create_account()
        logger.debug("Creating market wallet ... Ok!")

    def open_market_session(self):
        """
        If there are no market sessions, creates 1st session w/ default params
        Else, searches for last 'staged' session and opens it

        :return:
        """
        all_sessions = self.api.list_market_sessions()
        # Check if there are any sessions in the BD. If not, init first with
        # default parameters.
        if not all_sessions:
            logger.info("Creating first market session ...")
            first_session_cfg = settings.FirstSessionConfigs
            # Get create new market session:
            session = self.api.create_market_session(
                session_number=first_session_cfg.session_number,
                market_price=first_session_cfg.market_price,
                b_min=first_session_cfg.b_min,
                b_max=first_session_cfg.b_max,
                n_price_steps=first_session_cfg.n_price_steps,
                delta=first_session_cfg.delta
            )
            logger.info(session)
            logger.info("")

            # Post session weights if the session was correctly open:
            self.api.post_session_weights(
                session_id=session["market_session_id"],
                weights_p=first_session_cfg.weights_p
            )
            logger.info("Creating first market session ... Ok!")

        # List last market 'staged' sessions:
        staged_session = self.api.list_last_session(status='staged')
        logger.info(staged_session)
        logger.info("")

        # Change market session status from 'STAGED' to 'OPEN':
        self.api.update_market_session(
            session_id=staged_session["market_session_id"],
            status="open",
            open_ts=dt.datetime.utcnow()
        )

    def approve_buyers_bids(self):
        # Check open session:
        open_session = self.api.list_last_session(status='open')
        logger.info(open_session)
        logger.info("")

        # List bids for each session:
        bids = self.api.list_session_bids(
            session_id=open_session["market_session_id"]
        )
        logger.info(bids)
        logger.info("")

        # -- Get market wallet address:
        market_wallet_address = self.api.get_market_wallet_address()

        for b in bids:
            try:
                valid_in_tangle = self.tangle.validate_tangle_message(
                    message_id=b["tangle_msg_id"],
                    output_address=market_wallet_address,
                    expected_amount=b["max_payment"]
                )
                if valid_in_tangle:
                    rsp = self.api.post_validate_bid(
                        tangle_msg_id=b["tangle_msg_id"]
                    )
                    logger.info(rsp)
            except Exception as ex:
                logger.exception("Unable to place bid.")

    def close_market_session(self):
        # List last market 'staged' sessions:
        open_session = self.api.list_last_session(status='open')
        logger.info(open_session)
        logger.info("")

        # Change market session status from 'STAGED' to 'OPEN':
        self.api.update_market_session(
            session_id=open_session["market_session_id"],
            status="closed",
            close_ts=dt.datetime.utcnow()
        )

    def run_market_session(self):
        # todo: change this. right fixed to get always same measurements (.csv)
        market_launch_time = '2020-05-01 10:00:03.4536'
        market_launch_time = pd.to_datetime(market_launch_time).tz_localize(
            "UTC")
        market_launch_time = market_launch_time.to_pydatetime()

        # ################################
        # Fetch session data
        # #################################
        # Fetch session data:
        session_data, buyers_bids, active_sellers, price_weights = get_session_data(self.api)  # noqa

        # ###################################################
        # Check if there are sufficient bids to run market
        # ####################################################
        if len(buyers_bids) == 0:
            close_no_bids_session(
                api_controller=self.api,
                curr_session_data=session_data,
                curr_price_weights=price_weights
            )
            logger.error("No buyer bids available. "
                         "Finishing session & creating new one.")
            return False

        # ###################################
        # Convert units from IOTA to MIOTA:
        # ####################################
        session_data = convert_session_data_to_mi(data=session_data)
        buyers_bids = convert_buyers_bids_to_mi(bids=buyers_bids)

        # ################################
        # Check market buyers/sellers ID's
        # ################################
        buyers_ids = [x["user"] for x in buyers_bids if x["confirmed"] == True]
        sellers_ids = active_sellers

        # ################################
        # Query agents measurements:
        # ################################
        measurements = get_measurements_data(
            api_controller=self.api,
            buyers_ids=buyers_ids,
            sellers_ids=sellers_ids,
            market_launch_time=market_launch_time
        )

        # ################################
        # Create & Run Market Session
        # ################################
        mc = MarketClass(n_jobs=-1)
        mc.init_session(
            session_data=session_data,
            price_weights=price_weights,
            launch_time=market_launch_time
        )
        mc.show_session_details()
        mc.start_session(api_controller=self.api)
        # -- Load agents bids:
        mc.load_buyers_bids(bids=buyers_bids)
        mc.load_sellers(identifiers=active_sellers)
        # -- Load agents measurements data:
        mc.load_agents_measurements(measurements=measurements)
        # -- Run market session:
        mc.run_session()
        # -- Display session results
        mc.show_session_results()
        # -- Process payments:
        mc.process_payments(api_controller=self.api)
        # -- Update market price for next session:
        mc.update_market_price()
        # -- End session:
        mc.end_session(api_controller=self.api)
        # -- Open Next session:
        mc.open_next_session(api_controller=self.api)
        # -- Display session results
        mc.show_session_results()
        return True

    def transfer_tokens_out(self):
        # Important! There cant be open or running sessions, otherwise
        # users balance might change during this sessions and during
        # token transfer out. Leading to bad updates in database.
        # todo: improve this detection process in the future.
        open_sessions = self.api.list_market_sessions(status="open")
        if len(open_sessions) > 0:
            raise MarketSessionException("Failed to transfer tokens out. There are still sessions with 'open' status.")
        running_sessions = self.api.list_market_sessions(status="running")
        if len(running_sessions) > 0:
            raise MarketSessionException("Failed to transfer tokens out. There are still sessions with 'running' status.")

        # List of balances to transfer
        # Note: user must have balance > MINIMUM_WITHDRAW_AMOUNT (.env)
        balance_list = self.api.get_balances_to_transfer()
        logger.info(balance_list)
        logger.info("")

        # Transfer tokens out:
        # 1. Request user address (if non-existent, skips user)
        # 2. Transfer tokens to user & save node response:
        # 3. Validate transfer with Tangle Lookup:
        # 4. POST request to update users balance in database tables
        i = 0
        for b in balance_list:
            user_id = b["user"]
            balance_iota = int(b["balance"])
            try:
                address = self.api.get_user_wallet_address(user_id=user_id)
            except UserWalletException:
                logger.exception(f"Failed to get user {user_id} address.")
                continue

            try:
                node_response = self.wallet.transfer_tokens(
                    amount=balance_iota,
                    address=address
                )
                tangle_msg_id = node_response["id"]
            except Exception:
                logger.exception(f"Failed to transfer_tokens to {user_id}.")
                continue

            try:
                sleep(2)  # sleep a bit - let it solidify in tangle:
                self.tangle.validate_tangle_message(
                    message_id=tangle_msg_id,
                    expected_amount=balance_iota,
                    output_address=address,
                )
            except Exception:
                logger.exception(f"Failed to validate tangle_msg_id {tangle_msg_id}.")
                continue

            try:
                self.api.post_transfer_out(
                    user_id=user_id,
                    amount=balance_iota,
                    tangle_msg_id=tangle_msg_id,
                    user_wallet_address=address
                )
            except WalletTransferOutException:
                logger.exception(f"Failed to register tokens transfer out action.")
                continue
            i += 1