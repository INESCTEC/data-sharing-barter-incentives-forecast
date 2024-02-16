import fire

from time import time, sleep
from loguru import logger
from dotenv import load_dotenv

load_dotenv(".env")

from conf import settings
from src.MarketController import MarketController
from src.api.exception.APIException import NoMarketSessionException
from src.market.exception.ControllerException import PendingTransferOut


def retry(func, max_attempts=3, delay=1, exceptions=(Exception,)):
    for attempt in range(max_attempts):
        try:
            result = func()
            return result
        except exceptions as e:
            logger.debug(f"Attempt {attempt+1} failed:", e)
            if attempt < max_attempts - 1:
                logger.debug("Retrying after", delay, "seconds...")
                sleep(delay)
    raise Exception("Max attempts reached, could not get a valid result")


class MarketTasks(object):

    def __init__(self):
        self.open_session_retry_delay = 60
        self.open_session_retry_attempts = 15

    @staticmethod
    def approve_market_bids():
        """
        Script to approve buyers bids.
        Validates unconfirmed bids (DLT lookup) for the current market session.
        If there are no open market sessions, raises exception.
        """
        logger.info("-" * 79)
        t0 = time()
        msg_ = "Approving bids ..."
        logger.info(msg_)
        try:
            # Init market controller:
            market = MarketController()
            # Approve bids:
            market.approve_buyers_bids()
            logger.success(f"{msg_} Ok! {time() - t0:.2f}s")
        except Exception:
            logger.exception(f"{msg_} Failed! {time() - t0:.2f}s")

    @staticmethod
    def open_session():
        """
        Open a market session.
        If there are no market sessions, creates 1st session w/ default params
        Else, searches last 'staged' session and changes its status to 'open'
        Note that a new session will not be created if there are pending
        transfer out transactions (from market to users) which have direct
        include in the final agent market balance.
        """
        logger.info("-" * 79)
        t0 = time()
        msg_ = "Opening session ..."
        logger.info(msg_)
        # Init market controller:
        market = MarketController()
        try:
            # Attempt to open market session:
            market.open_market_session()
            logger.success(f"{msg_} Ok! {time() - t0:.2f}s")
        except NoMarketSessionException:
            # NoMarketSession exception is raised when there is no
            # staged session
            logger.error(f"{msg_} Failed! {time() - t0:.2f}s")
        except Exception:
            logger.exception(f"{msg_} Failed! {time() - t0:.2f}s")

    def run_session(self):
        """
        Run a market session.
        1. Closes current open market session ( stops bidding )
        2. Executes market session
        3. Transfers final balances back to users
        4. Attempts to validate transfers (DLT lookup)
        """
        logger.info("-" * 79)
        t0 = time()
        msg_ = "Running session ..."
        logger.info(msg_)

        try:
            # Init market controller:
            market = MarketController()

            # Final attempt to approve bids before closing session:
            market.approve_buyers_bids()

            # Close market session (no more bids):
            market.close_market_session()

            # Run market session:
            if settings.RUN_REAL_MARKET:
                market.run_market_session()
            else:
                market.run_fake_market_session()

            # Transfer balances:
            market.transfer_tokens_out()

            # Validate transfers:
            market.validate_tokens_transfer()

            # Try to open new market session
            # (status change from 'staged' to 'open')
            # Will fail until validate tokens transfer is successful as
            # we cannot open new sessions if previous balance transfers
            # were not successfully registered in the DLT
            retry(market.open_market_session,
                  max_attempts=self.open_session_retry_attempts,
                  delay=self.open_session_retry_delay,
                  exceptions=(PendingTransferOut,))

            logger.success(f"{msg_} Ok! {time() - t0:.2f}s")
        except Exception:
            logger.exception(f"{msg_} Failed! {time() - t0:.2f}s")

    @staticmethod
    def validate_transfer_out():
        """
        Validate pending transfer out transactions and update agent balances.
        """
        logger.info("-" * 79)
        t0 = time()
        msg_ = "Validating transfer out ..."
        try:
            # Init market controller:
            market = MarketController()
            # Validate transfers:
            market.validate_tokens_transfer()
            logger.success(f"{msg_} Ok! {time() - t0:.2f}s")
        except Exception:
            logger.exception(f"{msg_} Failed! {time() - t0:.2f}s")


if __name__ == '__main__':
    fire.Fire(MarketTasks)
