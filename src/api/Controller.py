import numpy as np
import datetime as dt

from time import time
from loguru import logger
from http import HTTPStatus

from conf import settings
from .Endpoint import *
from .RequestController import RequestController
from .exception.APIException import *


class Controller(RequestController):
    seller_role_id = 2
    config = {
        "N_REQUEST_RETRIES": settings.N_REQUEST_RETRIES,
        "RESTAPI_HOST": settings.RESTAPI_HOST,
        "RESTAPI_PORT": settings.RESTAPI_PORT,
    }

    def __init__(self):
        RequestController.__init__(self, self.config)
        self.access_token = ""

    def __check_if_token_exists(self):
        if self.access_token is None:
            e_msg = "Access token is not yet available. Login first."
            logger.error(e_msg)
            raise ValueError(e_msg)

    def set_access_token(self, token):
        self.access_token = token

    def __request_template(self,
                           endpoint_cls: Endpoint,
                           log_msg: str,
                           exception_cls,
                           data: dict = None,
                           params: dict = None,
                           url_params: list = None,
                           ) -> dict:
        self.__check_if_token_exists()
        t0 = time()
        rsp = self.request(
            endpoint=endpoint_cls,
            data=data,
            params=params,
            url_params=url_params,
            auth_token=self.access_token)
        # -- Inspect response:
        if rsp.status_code == HTTPStatus.OK:
            logger.debug(f"{log_msg} ... Ok! ({time() - t0:.2f})")
            return rsp.json()
        elif rsp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
            log_msg_ = f"{log_msg} ... Failed! ({time() - t0:.2f})"
            raise exception_cls(message=log_msg_,
                                errors={"message": "Internal Server Error."})
        else:
            log_msg_ = f"{log_msg} ... Failed! ({time() - t0:.2f})"
            logger.error(log_msg_ + f"\n{rsp.json()}")
            raise exception_cls(message=log_msg_, errors=rsp.json())

    def register(self, email, password, password_conf,
                 first_name, last_name, role):
        t0 = time()
        log_ = f"Registering user {email}"
        logger.debug(f"{log_} ...")
        payload = {
            "email": email,
            "password": password,
            "password_confirmation": password_conf,
            "first_name": first_name,
            "last_name": last_name,
            "role": role,
        }
        rsp = self.request(
            endpoint=Endpoint(register.POST, register.uri),
            data=payload
        )
        if rsp.status_code == HTTPStatus.OK:
            logger.debug(f"{log_} ... Ok! ({time() - t0:.2f})")
            return rsp
        elif rsp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
            log_msg = f"{log_} ... Failed! ({time() - t0:.2f})"
            raise LoginException(message=log_msg,
                                 errors={"message": "Internal Server Error."})
        else:
            log_msg = f"{log_} ... Failed! ({time() - t0:.2f})"
            logger.error(log_msg + f"\n{rsp.json()}")
            raise RegisterException(message=log_msg, errors=rsp.json())

    def login(self, email: str, password: str):

        t0 = time()
        log_ = f"Logging in user {email}"

        payload = {
            "email": email,
            "password": password
        }
        rsp = self.request(
            endpoint=Endpoint(login.POST, login.uri),
            data=payload
        )

        if rsp.status_code == HTTPStatus.OK:
            logger.debug(f"{log_} ... Ok! ({time() - t0:.2f})")
            self.access_token = rsp.json()['access']
        elif rsp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
            log_msg = f"{log_} ... Failed! ({time() - t0:.2f})"
            raise LoginException(message=log_msg,
                                 errors={"message": "Internal Server Error."})
        else:
            log_msg = f"{log_} ... Failed! ({time() - t0:.2f})"
            logger.error(log_msg + f"\n{rsp.json()}")
            raise LoginException(message=log_msg, errors=rsp.json())

    def get_user_wallet_address(self, user_id):
        params = {"user": user_id}
        response = self.__request_template(
            endpoint_cls=Endpoint(wallet_address.GET, wallet_address.uri),
            log_msg=f"Getting user {user_id} wallet address",
            params=params,
            exception_cls=UserWalletException
        )
        if len(response['data']) == 0:
            logger.error(f"No address found for user {user_id}")
            raise UserWalletException
        else:
            return response['data'][0]["wallet_address"]

    def get_balances_to_transfer(self):
        params = {"balance__gte": settings.MINIMUM_WITHDRAW_AMOUNT}
        response = self.__request_template(
            endpoint_cls=Endpoint(market_balance.GET, market_balance.uri),
            log_msg="Getting account balances to transfer (>1Mi)",
            params=params,
            exception_cls=MarketAccountException
        )
        return response['data']

    def get_user_market_balances(self, user_id=None):
        params = {}
        if user_id:
            params["user"] = user_id
        response = self.__request_template(
            endpoint_cls=Endpoint(market_balance.GET, market_balance.uri),
            log_msg="Getting account balances for market users",
            exception_cls=MarketAccountException
        )
        return response['data']

    def list_users(self):
        response = self.__request_template(
            endpoint_cls=Endpoint(user_list.GET, user_list.uri),
            log_msg="Getting users",
            exception_cls=UserException
        )
        return response['data']

    def create_market_session(self,
                              session_number: int,
                              market_price,
                              b_min,
                              b_max,
                              n_price_steps,
                              delta):
        payload = {
            "session_number": session_number,
            "market_price": market_price,
            "b_min": b_min,
            "b_max": b_max,
            "n_price_steps": n_price_steps,
            "delta": delta,
        }
        response = self.__request_template(
            endpoint_cls=Endpoint(market_session.POST, market_session.uri),
            log_msg="Creating market session",
            data=payload,
            exception_cls=MarketSessionException
        )
        return response['data']

    def update_market_session(self, session_id: int, **kwargs):
        # prepare kwargs:
        if isinstance(kwargs.get("launch_ts", None), dt.datetime):
            kwargs["launch_ts"] = kwargs["launch_ts"].strftime("%Y-%m-%dT%H:%M:%S.%f")  # noqa
        if isinstance(kwargs.get("finish_ts", None), dt.datetime):
            kwargs["finish_ts"] = kwargs["finish_ts"].strftime("%Y-%m-%dT%H:%M:%S.%f")  # noqa
        if isinstance(kwargs.get("close_ts", None), dt.datetime):
            kwargs["close_ts"] = kwargs["close_ts"].strftime("%Y-%m-%dT%H:%M:%S.%f")  # noqa
        if isinstance(kwargs.get("open_ts", None), dt.datetime):
            kwargs["open_ts"] = kwargs["open_ts"].strftime("%Y-%m-%dT%H:%M:%S.%f")  # noqa
        # -- Perform Request:
        payload = {}
        payload.update(kwargs)
        response = self.__request_template(
            endpoint_cls=Endpoint(market_session.PATCH, market_session.uri),
            log_msg=f"Updating market session {session_id}",
            data=payload,
            url_params=[session_id],
            exception_cls=MarketSessionException
        )
        return response['data']

    def post_session_weights(self, session_id: int, weights_p):
        if not isinstance(weights_p, list):
            weights_p = list(weights_p)
        payload = {
            "weights_p": weights_p,
            "market_session": session_id,
        }
        response = self.__request_template(
            endpoint_cls=Endpoint(market_price_weight.POST,
                                  market_price_weight.uri),
            log_msg="Posting session weights",
            data=payload,
            exception_cls=MarketSessionException
        )
        return response['data']

    def list_market_sessions(self, status=None):
        params = {}
        if status is not None:
            params["status"] = status
        response = self.__request_template(
            endpoint_cls=Endpoint(market_session.GET, market_session.uri),
            log_msg="Getting market sessions",
            params=params,
            exception_cls=MarketSessionException
        )
        return response['data']

    def list_last_session(self, status: str = None):
        # todo: ir logo buscar só uma sessao pela rest (e.g. query limit 1)
        params = {"latest_only": True}
        if status is not None:
            params["status"] = status
            msg = f"Getting last '{status}' market session"
        else:
            msg = "Getting last market session."

        response = self.__request_template(
            endpoint_cls=Endpoint(market_session.GET, market_session.uri),
            log_msg=msg,
            params=params,
            exception_cls=MarketSessionException
        )
        # Get sessions data - check if there are open sessions:
        sessions = response['data']
        if len(sessions) == 0:
            log_msg = "No market sessions available."
            logger.error(log_msg)
            raise NoMarketSessionException(message=log_msg,
                                           errors=response)
        else:
            return sessions[0]

    def list_session_weights(self, session_id: int):
        params = {"market_session": session_id}
        response = self.__request_template(
            endpoint_cls=Endpoint(market_price_weight.GET,
                                  market_price_weight.uri),
            log_msg=f"Getting weights for session {session_id}",
            params=params,
            exception_cls=MarketSessionException
        )
        weights_ = [(x["id"], x["weights_p"]) for x in response['data']]
        sorted_weights_ = sorted(weights_, key=lambda tup: tup[1])
        return np.array([x[1] for x in sorted_weights_])

    def list_user_resources(self, to_forecast=None):
        params = {}
        if to_forecast is not None:
            params["to_forecast"] = to_forecast
        response = self.__request_template(
            endpoint_cls=Endpoint(user_resources.GET, user_resources.uri),
            log_msg=f"Getting user resources (to_forecast = {to_forecast})",
            params=params,
            exception_cls=UserException
        )
        return response["data"]

    def place_bid(self,
                  session_id: int,
                  resource_id: int,
                  bid_price,
                  max_payment,
                  gain_func):

        payload = {
            "market_session": session_id,
            "resource": resource_id,
            "bid_price": bid_price,
            "max_payment": max_payment,
            "gain_func": gain_func
        }
        response = self.__request_template(
            endpoint_cls=Endpoint(market_bid.POST, market_bid.uri),
            log_msg=f"Posting bid for market session ID: {session_id}",
            data=payload,
            exception_cls=MarketBidException
        )
        return response['data']

    def get_market_wallet_address(self):
        response = self.__request_template(
            endpoint_cls=Endpoint(market_wallet_address.GET,
                                  market_wallet_address.uri),
            log_msg="Getting market wallet address",
            exception_cls=MarketWalletAddressException
        )
        return response['data']["wallet_address"]

    def register_market_wallet_address(self, address):
        payload = {
            "wallet_address": address,
        }
        response = self.__request_template(
            endpoint_cls=Endpoint(market_wallet_address.POST,
                                  market_wallet_address.uri),
            log_msg="Getting market wallet address",
            data=payload,
            exception_cls=MarketWalletAddressException
        )
        return response['data']

    def update_market_wallet_address(self, new_address):
        payload = {
            "wallet_address": new_address,
        }
        response = self.__request_template(
            endpoint_cls=Endpoint(market_wallet_address.PUT,
                                  market_wallet_address.uri),
            log_msg="Updating market wallet address",
            data=payload,
            exception_cls=MarketWalletAddressException
        )
        return response['data']

    def list_session_bids(self,
                          session_id: int,
                          confirmed: int = None):
        params = {"market_session": session_id}
        if confirmed is not None:
            params["confirmed"] = confirmed
        response = self.__request_template(
            endpoint_cls=Endpoint(market_bid.GET, market_bid.uri),
            log_msg=f"Listing bids for session {session_id}",
            params=params,
            exception_cls=MarketSessionException
        )
        return response["data"]

    def post_validate_bid(self, tangle_msg_id):
        payload = {"tangle_msg_id": tangle_msg_id}
        response = self.__request_template(
            endpoint_cls=Endpoint(market_validate_bids.POST,
                                  market_validate_bids.uri),
            log_msg=f"Validating Tangle Message ID {tangle_msg_id}",
            data=payload,
            exception_cls=MarketSessionException
        )
        return response["data"]

    def list_pending_transfer_out(self, user_id=None):
        params = {
            "is_solid": False
        }
        if user_id:
            params["user"] = user_id
        response = self.__request_template(
            endpoint_cls=Endpoint(market_transfer_out.GET,
                                  market_transfer_out.uri),
            log_msg="Listing pending transfers",
            params=params,
            exception_cls=WalletTransferOutException
        )
        return response["data"]

    def post_transfer_out(self, user_id, amount, tangle_msg_id,
                          user_wallet_address):
        payload = {
            "user": user_id,
            "amount": amount,
            "tangle_msg_id": tangle_msg_id,
            "user_wallet_address": user_wallet_address,
        }
        response = self.__request_template(
            endpoint_cls=Endpoint(market_transfer_out.POST,
                                  market_transfer_out.uri),
            log_msg=f"Registering wallet transfer out tokens action - Tangle message ID: {tangle_msg_id}",
            data=payload,
            exception_cls=WalletTransferOutException
        )
        return response["data"]

    def put_confirm_transfer_out(self, withdraw_transfer_id, is_solid):
        payload = {
            "withdraw_transfer_id": withdraw_transfer_id,
            "is_solid": is_solid,
        }
        response = self.__request_template(
            endpoint_cls=Endpoint(market_transfer_out.PUT,
                                  market_transfer_out.uri),
            log_msg=f"Updating wallet transfer out - transfer_id: {withdraw_transfer_id}",
            data=payload,
            exception_cls=WalletTransferOutException
        )
        return response["data"]

    def post_session_market_fee(self,
                                session_id: int,
                                fee_amount: float):
        payload = {
            "market_session": session_id,
            "amount": fee_amount,
        }
        response = self.__request_template(
            endpoint_cls=Endpoint(market_session_fee.POST,
                                  market_session_fee.uri),
            log_msg=f"Registering market fee of {fee_amount} "
                    f"for market session {session_id}",
            data=payload,
            exception_cls=MarketSessionFee
        )
        return response['data']

    def post_session_balance(self,
                             user_id: int,
                             resource_id: int,
                             amount: int,
                             transaction_type: str,
                             session_id: int):
        payload = {
            "amount": amount,
            "user": user_id,
            "resource": resource_id,
            "transaction_type": transaction_type,
            "market_session": session_id,
        }
        response = self.__request_template(
            endpoint_cls=Endpoint(market_session_balance.POST,
                                  market_session_balance.uri),
            log_msg=f"Transferring {amount} to user {user_id} market account",
            data=payload,
            exception_cls=UserSessionBalance
        )
        return response['data']
