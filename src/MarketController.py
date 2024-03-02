import json
import pandas as pd
import datetime as dt

from loguru import logger
from collections import defaultdict

from conf import settings

from payment.PaymentGateway.IOTAPayment.IOTAClientController import IOTAClientController  # noqa
from payment.PaymentGateway.IOTAPayment.IOTAClientController import MultipleOutputSchema, TransactionOutput  # noqa

from .api import Controller
from .wallet.WalletController import WalletController
from .market import MarketClass
from .market.helpers.api_helpers import (
    get_session_data,
    close_no_bids_session
)
from .market.helpers.db_helpers import (
    get_measurements_data,
    get_measurements_data_mock,
)
from .market.helpers.units_helpers import (
    convert_session_data_to_mi,
    convert_buyers_bids_to_mi,
)

from .api.exception.APIException import *
from .market.exception.ControllerException import PendingTransferOut


class MarketController:
    def __init__(self):
        # Market Wallet Controller:
        self.wallet = WalletController()
        # @todo: @Andre - revisit WalletController, it is forcing automatic
        #   wallet creation.
        # Tangle Controller:
        self.tangle = IOTAClientController(node_url=[settings.IOTA_NODE_URL])
        # Market API Controller:
        self.api = Controller()
        self.api.login(email=settings.MARKET_EMAIL,
                       password=settings.MARKET_PASSWORD)

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
            logger.info(f"Validating bid {b['id']} - {b['tangle_msg_id']}")

            if b["tangle_msg_id"] is None:
                logger.error(f"Bid {b['id']} does not have a tangle_msg_id.")
                continue

            try:
                valid_in_tangle = self.tangle.validate_transaction_id(
                    transaction_id=b["tangle_msg_id"],
                    address=market_wallet_address,
                    amount=b["max_payment"]
                )

                if valid_in_tangle:
                    rsp = self.api.post_validate_bid(
                        tangle_msg_id=b["tangle_msg_id"]
                    )
                    logger.info(f"Validating bid {b['id']} - "
                                f"{b['tangle_msg_id']} ... Ok!")
                    logger.debug(rsp)

            except Exception:
                logger.exception(f"Validating bid {b['id']} - "
                                 f"{b['tangle_msg_id']} ... Failed!")

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
        bids_per_resource = convert_buyers_bids_to_mi(bids=bids_per_resource)

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
        mc = MarketClass(n_jobs=settings.N_JOBS,
                         auto_feature_selection=False,
                         auto_feature_engineering=True,
                         enable_db_uploads=True)
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
            close_no_bids_session(
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
                    'tangle_msg_id': 'xaxxxxsaxacas',
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

        # ###################################
        # Convert units from IOTA to MIOTA:
        # ####################################
        session_data = convert_session_data_to_mi(data=session_data)
        bids_per_resource = convert_buyers_bids_to_mi(bids=bids_per_resource)

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
        Transfer current balances (IOTA tokens) back to each user wallet

        :return:
        """

        # Important! There cant be open or running sessions, otherwise
        # users balance might change during this sessions and during
        # token transfer out. Leading to bad updates in database.
        # todo: improve this detection process in the future.
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

        # Prepare multi-transfer operations:
        transfer_list = []
        total_transfer = 0
        for b in balance_list:
            user_id = b["user"]
            balance_iota = int(b["balance"])
            try:
                address = self.api.get_user_wallet_address(user_id=user_id)
                b["address"] = address
                transfer_list.append({
                    "amount": balance_iota,
                    "address": address
                })
                total_transfer += balance_iota
            except UserWalletException:
                logger.error(f"Failed to get user {user_id} wallet address.")
                continue

        if len(transfer_list) == 0:
            log_msg_ = "Balance transfer-out list is empty."
            logger.warning(log_msg_)
            return False

        # Market balance:
        balance = self.wallet.get_balance()
        balance = int(balance["baseCoin"]["available"])
        logger.info(f"Current balance (market wallet): {balance / 1000000}Mi")
        logger.info(f"Total to transfer: {total_transfer / 1000000}Mi")
        logger.info(f"Expected remaining: {(balance - total_transfer) / 1000000}Mi")

        # Check if market wallet has sufficient funds to transfer:
        amount_to_transfer = sum([x["amount"] for x in transfer_list])
        if balance < amount_to_transfer:
            log_msg_ = (f"Insufficient funds in market wallet ({balance}) "
                        f"to allow a {amount_to_transfer} transfer.")
            logger.error(log_msg_)
            return False

        try:
            # Create multi-transfer operations:
            logger.info("Performing multi-output txn ...")
            node_response = self.wallet.transfer_tokens_multi_address(
                transfer_list=transfer_list
            )
            tangle_msg_id = node_response.transactionId
            logger.debug(f"Tangle Message ID: {tangle_msg_id}")
            logger.success("Performing multi-output txn ... Ok!")
        except Exception:
            logger.exception("Unexpected transfer failure!")
            logger.error("Performing multi-output txn ... Failed!")
            return False

        # Register each transfer in DB:
        for b in balance_list:
            user_id = b["user"]
            balance_iota = int(b["balance"])
            address = b["address"]
            try:
                transfer_data = self.api.post_transfer_out(
                    user_id=user_id,
                    amount=balance_iota,
                    tangle_msg_id=tangle_msg_id,
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

        transfers_by_msg_id = defaultdict(list)
        for ttx in transfer_list:
            tangle_msg_id = ttx["tangle_msg_id"]
            transfer_data = {
                "address": ttx["user_wallet_address"],
                "amount": ttx["amount"],
                "withdraw_transfer_id": ttx["withdraw_transfer_id"]
            }
            transfers_by_msg_id[tangle_msg_id].append(transfer_data)

        all_successful = True
        for tangle_msg_id, transfer_list in transfers_by_msg_id.items():

            tx_outputs = [TransactionOutput(id=x["withdraw_transfer_id"],
                                            address=x["address"],
                                            amount=x["amount"])
                          for x in transfer_list]

            multiple_output = MultipleOutputSchema(
                transaction_id=tangle_msg_id,
                transactions=tx_outputs)

            try:
                # Validate message ID:
                self.tangle.validate_multiple_outputs(multiple_output)
            except Exception:
                logger.exception("Unexpected validation failure!")
                all_successful = False
                continue

            valid_tx = all([x.confirmed for x in tx_outputs])

            if valid_tx:
                logger.success(f"Transfer output Txn {tangle_msg_id} is valid!")  # noqa
                for tid in transfer_list:
                    try:
                        response = self.api.put_confirm_transfer_out(
                            withdraw_transfer_id=tid["withdraw_transfer_id"],
                            is_solid=True,
                        )
                        logger.debug(f"Transfer out response: {response}")
                    except WalletTransferOutException:
                        logger.error(f"Failed to register transfer out "
                                     f"operation for withdraw ID: {tid['withdraw_transfer_id']}")
                        continue
            else:
                logger.error(f"Transfer output Txn {tangle_msg_id} is invalid!")  # noqa
                all_successful = False

        return all_successful
