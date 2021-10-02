
import os

from dotenv import load_dotenv

__ENV_PATH__ = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(__ENV_PATH__)

from core.MarketController import MarketController
from core.api.exception.APIException import NoMarketSessionException

market = MarketController()

try:
    # Create first market session:
    market.open_market_session()
except NoMarketSessionException:
    pass

cf = input("A new market session is open. Place your bids.\n"
           "After, confirm to continue to bid approval.\n"
           "> Confirm (Y/n)")
if cf.lower() not in ["y", "n"]:
    raise IOError("Input error. Valid inputs are y-yes or n-no")

# Approve buyers bids:
market.approve_buyers_bids()

input("Press any key to continue.")
# Close market session (no more bids):
market.close_market_session()

input("Press any key to continue.")
# Run market session:
market.run_market_session()

input("Press any key to continue.")
# Transfer tokens back to clients:
market.transfer_tokens_out()
