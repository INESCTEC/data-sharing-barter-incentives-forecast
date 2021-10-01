"""
Description:
- Logs in using market credentials
- Creates market roles (in VALOREM platform)
- Creates market wallet (in VALOREM platform)
- Opens the first market session

Date: 2021-06-18
Author: Ricardo Andrade (jose.r.andrade@inesctec.pt)
"""

import os

from loguru import logger
from dotenv import load_dotenv

__ENV_PATH__ = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(__ENV_PATH__)

from core.wallet import WalletController


def create_market_wallet():
    logger.info("Creating market wallet ...")
    # Initialize API controller:
    wallet = WalletController()
    wallet.create_wallet(store_mnemonic=True)
    wallet.create_account()
    logger.info("Creating market wallet ... Ok!")


if __name__ == '__main__':
    create_market_wallet()
