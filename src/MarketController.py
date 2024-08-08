import json
import pandas as pd
import datetime as dt

from loguru import logger

from conf import settings

from .api import Controller
from .wallet.WalletController import WalletController
from .market import MarketClass
from .market.helpers.api_helpers import (
    get_session_data,
    close_failed_session
)
from .market.helpers.db_helpers import (
    get_measurements_data,
    get_measurements_data_mock,
)
from .api.exception.APIException import *
from .market.exception.ControllerException import PendingTransferOut


class MarketController:
    def __init__(self):
        # Market API Controller:
        self.api = Controller()
        self.api.login(email=settings.MARKET_EMAIL,
                       password=settings.MARKET_PASSWORD)
        self.payment_type = self.api.get_market_payment_processor()["base_unit"]
        self.payment_type = "IOTA" if self.payment_type == "SMR" else "ERC20"
        # Market Wallet Controller:
        self.wallet = WalletController(payment_type=self.payment_type)

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
                session_id=session["id"],
                weights_p=first_session_cfg.weights_p
            )
            logger.info("Creating first market session ... Ok!")

        # List last market 'staged' sessions:
        staged_session = self.api.list_last_session(status='staged')
        logger.info("Current 'STAGED' session:")
        logger.info(staged_session)
        logger.info("")

        # Do not allow opening a new market session unless all the market
        # transfer out transactions (which return users balance to each user)
        # are valid
        pending_transfer_list = self.api.list_pending_transfer_out()
        if len(pending_transfer_list) > 0:
            raise PendingTransferOut(
                message=f"Unable to open new session as "
                        f"there are still {len(pending_transfer_list)} "
                        f"unconfirmed market transfers from last session. "
                        f"Please validate token transfers first.")

        # Change market session status from 'STAGED' to 'OPEN':
        self.api.update_market_session(
            session_id=staged_session["id"],
            status="open",
            open_ts=dt.datetime.utcnow()
        )

    def register_market_wallet_address(self, address):
        """
        Register new market wallet address

        :param str address: Market wallet address
        :return:
        """
        response = self.api.register_market_wallet_address(address=address)
        logger.info(response)
        logger.info("")
        return response

    def get_market_wallet_address(self):
        """
        Request Market wallet address

        :return:
        """
        response = self.api.get_market_wallet_address()
        logger.info(response)
        logger.info("")
        return response

    def update_market_wallet_address(self, new_address):
        """
        Update current market wallet address

        :param str new_address: New address to replace current market address
        :return:
        """
        response = self.api.update_market_wallet_address(
            new_address=new_address
        )
        logger.info(response)
        logger.info("")
        return response

    def get_buyers_bids(self):
        """
        Request buyers bids for last 'open' session

        :return:
        """
        # Check open session:
        latest_session = self.api.list_last_session()
        logger.info("Latest session:")
        logger.info(latest_session)
        logger.info("")
        # List bids for each session:
        bids = self.api.list_session_bids(session_id=latest_session["id"])
        logger.info(f"There are {len(bids)} for this session.")
        logger.info(json.dumps(bids, indent=2))
        logger.info("")
        return bids

    def list_last_session(self):
        """
        Request buyers bids for last 'open' session

        :return:
        """
        # Check open session:
        session = self.api.list_last_session()
        logger.info("Last session available:")
        logger.info(json.dumps(session, indent=2))
        logger.info("")
        return session

    def set_session_status(self, session_id, new_status):
        """
        Request buyers bids for last 'open' session

        :return:
        """
        status = self.api.update_market_session(
            session_id=session_id,
            status=new_status
        )
        logger.info(json.dumps(status, indent=2))
        logger.info("")

    def return_rejected_bids(self):
        pass

    def approve_buyers_bids(self):
        """
        Approve buyers bids for current session

        :return:
        """
        # Check current open session:
        open_session = self.api.list_last_session(status='open')
        logger.info("Current 'OPEN' session:")
        logger.info(open_session)
        logger.info("")

        # List bids for each session:
        bids = self.api.list_session_bids(
            session_id=open_session["id"],
            confirmed=False,
        )
        logger.info(f"There are {len(bids)} 'UNCONFIRMED' bids for this "
                    f"session.")
        logger.info(json.dumps(bids, indent=2))
        logger.info("")

        # -- Get market wallet address:
        market_wallet_address = self.api.get_market_wallet_address()

        for b in bids:
            logger.info(f"Validating bid {b['id']} - {b['transaction_id']}")

            if b["transaction_id"] is None:
                logger.error(f"Bid {b['id']} does not have a transaction_id.")
                continue

            try:
                valid_txn = self.wallet.validate_transaction_id(
                    transaction_id=b["transaction_id"],
                    to_address=market_wallet_address,
                    amount=b["max_payment"]
                )

                if valid_txn:
                    rsp = self.api.post_validate_bid(
                        transaction_id=b["transaction_id"]
                    )
                    logger.info(f"Validating bid {b['id']} - "
                                f"{b['transaction_id']} ... Ok!")
                    logger.debug(rsp)

            except Exception as ex:
                logger.error(ex)
                logger.exception(f"Validating bid {b['id']} - "
                                 f"{b['transaction_id']} ... Failed!")

    def close_market_session(self):
        """
        Close current 'OPEN' market session

        :return:
        """
        # List last market 'open' sessions:
        open_session = self.api.list_last_session(status='open')
        logger.info("Current 'OPEN' session:")
        logger.info(open_session)
        logger.info("")

        # Change market session status from 'OPEN' to 'CLOSED':
        self.api.update_market_session(
            session_id=open_session["id"],
            status="closed",
            close_ts=dt.datetime.utcnow()
        )

    def run_market_session(self):
        """
        Run last 'closed' market session. Session state is updated to
        'running' during execution and to 'finished' once it is complete.

        :return:
        """
        launch_time = dt.datetime.utcnow()
        launch_time = pd.to_datetime(launch_time).tz_localize("UTC")
        launch_time = launch_time.to_pydatetime()

        # ################################
        # Fetch session info
        # #################################
        # Fetch session info:
        session_info = get_session_data(self.api)
        session_data = session_info["session_data"]
        bids_per_resource = session_info["bids_per_resource"]
        users_resources = session_info["users_resources"]
        price_weights = session_info["price_weights"]
        logger.debug(f"\nSession info:"
                     f"\n{json.dumps(users_resources, indent=3)}")

        # ###################################################
        # Check if there are sufficient bids to run market
        # ####################################################
        if len(bids_per_resource) == 0:
            close_failed_session(
                api_controller=self.api,
                curr_session_data=session_data,
                curr_price_weights=price_weights
            )
            logger.error("No buyer bids available. "
                         "Finishing session & creating new one.")
            return False

        # ################################
        # Query agents measurements:
        # ################################
        measurements = get_measurements_data(
            users_resources=users_resources,
            market_launch_time=launch_time
        )

        # ################################
        # Create & Run Market Session
        # ################################
        try:
            mc = MarketClass(n_jobs=settings.N_JOBS,
                             auto_feature_selection=False,
                             auto_feature_engineering=True,
                             enable_db_uploads=False)
            mc.init_session(
                session_data=session_data,
                price_weights=price_weights,
                launch_time=launch_time
            )
            mc.show_session_details()
            mc.start_session(api_controller=self.api)
            # -- Load resources bids:
            mc.load_users_resources(users_resources=users_resources)
            mc.load_resources_bids(bids=bids_per_resource)
            # -- Load resources measurements data:
            mc.load_resources_measurements(measurements=measurements)
            # -- Run market session:
            mc.define_payments_and_forecasts()
            mc.define_sellers_revenue()
            mc.save_session_results()
            mc.validate_session_results(raise_exception=True)
            # -- Display session results
            mc.show_session_results()
            # -- Process payments:
            mc.process_payments(api_controller=self.api,
                                wallet_controller=self.wallet)
            # -- Update market price for next session:
            mc.update_market_price()
            # -- End session:
            mc.end_session(api_controller=self.api)
            # -- Open Next session:
            mc.open_next_session(api_controller=self.api)
            # -- Display session results
            mc.show_session_results()
            return True
        except BaseException as ex:
            logger.error(ex)
            logger.exception("Failed to run session. Closed and staged new one.")
            close_failed_session(
                api_controller=self.api,
                curr_session_data=session_data,
                curr_price_weights=price_weights
            )
            return False

    def run_fake_market_session(self):
        """
        Run last 'closed' market session. Session state is updated to
        'running' during execution and to 'finished' once it is complete.

        :return:
        """
        from copy import deepcopy
        launch_time = dt.datetime.utcnow()
        launch_time = pd.to_datetime(launch_time).tz_localize("UTC")
        launch_time = launch_time.to_pydatetime()

        # ################################
        # Fetch session info
        # #################################
        # Fetch session info:
        session_info = get_session_data(self.api)
        logger.debug(f"\nSession data:"
                     f"\n{session_info}")
        session_data = session_info["session_data"]
        bids_per_resource = session_info["bids_per_resource"]
        users_resources = session_info["users_resources"]
        price_weights = session_info["price_weights"]
        logger.debug(f"\nUser resources (before fake users):"
                     f"\n{json.dumps(users_resources, indent=3)}")

        # ###################################################
        # Check if there are sufficient bids to run market
        # ####################################################
        if len(bids_per_resource) == 0:
            close_failed_session(
                api_controller=self.api,
                curr_session_data=session_data,
                curr_price_weights=price_weights
            )
            logger.error("No buyer bids available. "
                         "Finishing session & creating new one.")
            return False
        # elif len(bids_per_resource) > 1:
        #     logger.error("You cannot have more than 1 bid while on "
        #                  "'fake' market mode.")
        #     return False
        # else:
        #     if len(users_resources) > 1:
        #         logger.error("You cannot have more than 1 resource registered "
        #                      "in the market, in this 'fake' market mode.")
        #         return False

        resources_w_bids = set([x["resource"] for x in bids_per_resource])
        users_w_bids = set([x["user"] for x in bids_per_resource])
        bid_id_list = set([x["id"] for x in bids_per_resource])
        _last_res = max(resources_w_bids) + 1
        _last_user = max(users_w_bids) + 1
        _last_bid_id = max(bid_id_list) + 1
        _n = 5  # number of extra resources/users/bids
        extra_resources = [x for x in range(_last_res, _last_res + _n)]
        extra_users = [x for x in range(_last_user, _last_user + _n)]
        extra_bid_ids = [x for x in range(_last_bid_id, _last_bid_id + _n)]

        zip_gen = zip(extra_resources, extra_users, extra_bid_ids)
        for (res_id, user_id, bid_id) in zip_gen:
            bids_per_resource.append(
                {
                    'id': bid_id,
                    'transaction_id': 'xaxxxxsaxacas',
                    'max_payment': session_data["market_price"],
                    'bid_price': session_data["market_price"],
                    'gain_func': 'mse',
                    'confirmed': True,
                    'registered_at': '2022-01-04T10:32:15.376562Z',
                    'has_forecasts': True,
                    'user': user_id,
                    'resource': res_id,
                    'market_session': session_data["id"]
                }
            )
            users_resources.append(
                {'id': res_id,
                 'name': f'user-{user_id}-resource-{res_id}',
                 'type': 'measurements',
                 'to_forecast': True,
                 'registered_at': '2022-01-04T10:31:32.785753Z',
                 'user': user_id}
            )

        logger.debug(f"\nUser resources (after fake users):"
                     f"\n{json.dumps(users_resources, indent=3)}")

        # ################################
        # Query agents measurements:
        # ################################
        measurements = get_measurements_data_mock(
            users_resources=users_resources,
            market_launch_time=launch_time
        )

        # ################################
        # Create & Run Market Session
        # ################################
        mc = MarketClass(n_jobs=settings.N_JOBS,
                         enable_db_uploads=True,
                         auto_feature_selection=False)
        mc.init_session(
            session_data=session_data,
            price_weights=price_weights,
            launch_time=launch_time
        )
        mc.show_session_details()
        mc.start_session(api_controller=self.api)
        # -- Load resources bids:
        mc.load_users_resources(users_resources=users_resources)
        mc.load_resources_bids(bids=bids_per_resource)
        # -- Load resources measurements data:
        mc.load_resources_measurements(measurements=measurements)
        # -- Run market session:
        mc.define_payments_and_forecasts()
        mc.define_sellers_revenue()
        mc.save_session_results()
        # -- Display session results
        mc.show_session_results()

        # Remove fictitious agents / resources
        for res in extra_resources:
            del mc.sellers_data[res]
            del mc.buyers_data[res]
            del mc.mkt_sess.market_fee_per_resource[res]
            del mc.mkt_sess.buyers_results[res]
            del mc.mkt_sess.sellers_results[res]

        # Reset market fees (to one resource only)
        mc.mkt_sess.total_market_fee = sum(mc.mkt_sess.market_fee_per_resource.values())

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

    def list_user_market_balance(self):
        """
        List current market balance for every user registered in the market

        :return:
        """
        balance_list = self.api.get_user_market_balances()
        logger.info(json.dumps(balance_list, indent=2))
        logger.info("")

    def transfer_tokens_out(self):
        """
        Transfer current balances back to each user wallet

        :return:
        """

        # Important! There cant be open or running sessions, otherwise
        # users balance might change during this sessions and during
        # token transfer out. Leading to bad updates in database.
        open_sessions = self.api.list_market_sessions(status="open")
        if len(open_sessions) > 0:
            log_msg_ = ("Can only perform transfer-out operations while "
                        "there are no sessions in 'open' status.")
            raise WalletTransferOutException(
                message=log_msg_,
                errors={"message": log_msg_}
            )
        running_sessions = self.api.list_market_sessions(status="running")
        if len(running_sessions) > 0:
            log_msg_ = ("Can only perform transfer-out operations while "
                        "there are no sessions in 'running' status.")
            raise WalletTransferOutException(
                message=log_msg_,
                errors={"message": log_msg_}
            )

        # List of balances to transfer
        # Note: user must have balance > MINIMUM_WITHDRAW_AMOUNT (.env)
        balance_list = self.api.get_balances_to_transfer()
        balance_list = [x for x in balance_list if x["user"] != 1]  # todo: remove based on role / admin
        logger.info(balance_list)
        logger.info("")

        # Get market wallet address:
        market_address = self.wallet.get_address()

        # Prepare multi-transfer operations:
        transfer_list = []
        total_transfer = 0
        for b in balance_list:
            user_id = b["user"]
            balance_ = self.wallet.base_to_transaction_units(b["balance"])
            try:
                address = self.api.get_user_wallet_address(user_id=user_id)
                b["address"] = address
                transfer_list.append({
                    "user_id": user_id,
                    "amount": balance_,
                    "from_identifier": market_address,
                    "to_identifier": address
                })
                total_transfer += balance_
            except UserWalletException:
                logger.error(f"Failed to get user {user_id} wallet address.")
                continue

        if len(transfer_list) == 0:
            log_msg_ = "Balance transfer-out list is empty."
            logger.warning(log_msg_)
            return False

        # Market balance:
        wallet_balance, unit = self.wallet.get_balance()
        logger.info(f"Current balance (market wallet): {wallet_balance}{unit}")
        logger.info(f"Total to transfer: {total_transfer}{unit}")
        logger.info(f"Expected remaining: {(wallet_balance - total_transfer)}{unit}")

        # Check if market wallet has sufficient funds to transfer:
        if wallet_balance < total_transfer:
            log_msg_ = (f"Insufficient funds in market wallet ({wallet_balance}) "
                        f"to allow a {total_transfer} transfer.")
            logger.error(log_msg_)
            return False

        try:
            # Create multi-transfer operations:
            logger.info("Performing multi-output txn ...")
            transactions = self.wallet.transfer_tokens_multi_address(
                transfer_list=transfer_list
            )
            logger.success("Performing multi-output txn ... Ok!")
        except Exception as ex:
            logger.error(ex)
            logger.exception("Unexpected transfer failure!")
            logger.error("Performing multi-output txn ... Failed!")
            return False

        # Register each transfer in DB:
        for txn in transactions:
            user_id = txn["user_id"]
            amount = txn["amount"]
            address = txn["to_identifier"]
            transaction_id = txn["transaction_id"]
            amount = self.wallet.transaction_to_base_units(amount)
            try:
                transfer_data = self.api.post_transfer_out(
                    user_id=user_id,
                    amount=amount,
                    transaction_id=transaction_id,
                    user_wallet_address=address
                )
                logger.debug(transfer_data)
            except WalletTransferOutException:
                logger.exception("Failed to register tokens transfer out action.")
                continue

    def validate_tokens_transfer(self):
        """
        Validate all balance transfers and update its state in the platform

        :return:
        """
        transfer_list = self.api.list_pending_transfer_out()

        if len(transfer_list) == 0:
            log_msg_ = "No pending transfer out operations."
            logger.warning(log_msg_)
            return

        all_successful = True
        for tid in transfer_list:
            valid_txn = self.wallet.validate_transaction_id(
                transaction_id=tid["transaction_id"],
                to_address=tid["user_wallet_address"],
                amount=tid["amount"]
            )

            if valid_txn:
                logger.success(f"Transfer output Txn {tid['transaction_id']} is valid!")  # noqa
                try:
                    response = self.api.put_confirm_transfer_out(
                        withdraw_transfer_id=tid["withdraw_transfer_id"],
                        confirmed=True,
                    )
                    logger.debug(f"Transfer out response: {response}")
                except WalletTransferOutException:
                    logger.error(f"Failed to register transfer out "
                                 f"operation for withdraw ID: {tid['withdraw_transfer_id']}")
                    continue

            else:
                logger.error(f"Transfer output Txn {tid['transaction_id']} is invalid!")  # noqa
                all_successful = False

        return all_successful
