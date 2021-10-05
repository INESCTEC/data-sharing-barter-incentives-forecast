
import os

from loguru import logger
from dotenv import load_dotenv

load_dotenv(".env")

from core.MarketController import MarketController
from core.api.exception.APIException import NoMarketSessionException


# logger:
format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}"
logger.add("files/logfile.log", format=format, level='DEBUG', backtrace=True)
logger.info("-" * 79)


market = MarketController()
#
# try:
#     # Create first market session:
#     market.open_market_session()
# except NoMarketSessionException:
#     pass
#
# cf = input("A new market session is open. Place your bids.\n"
#            "After, confirm to continue to bid approval.\n"
#            "> Confirm (Y/n)")
# if cf.lower() not in ["y", "n"]:
#     raise IOError("Input error. Valid inputs are y-yes or n-no")
#
# # Approve buyers bids:
# market.approve_buyers_bids()
#
# input("Press any key to continue.")
# # Close market session (no more bids):
# market.close_market_session()
#
# input("Press any key to continue.")
# # Run market session:
# market.run_market_session()

# Transfer tokens back to clients:
market.transfer_tokens_out()
# market.validate_tokens_transfer()


