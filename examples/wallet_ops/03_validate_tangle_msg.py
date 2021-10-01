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
from dotenv import dotenv_values

from market.wallet import TangleController

__ENV_PATH__ = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# logger:
format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}"
logger.add("files/logfile.log", format=format, level='DEBUG', backtrace=True)
logger.info("-" * 79)

# Configs:
config = dotenv_values(__ENV_PATH__)
logger.info(f".env\n{config}")

# Configs:
MESSAGE_IDENTIFIER = "466c51d80de510671b7f09380982ebb3bcb823606b528f8be0f699d4c6a69d9b"
OUTPUT_ADDRESS = "atoi1qrya4twypxes5sdt5nfge45txwwnzjuns899j5gj85lv732h33w5vmfa7zd"
EXPECTED_AMOUNT = 1000000

# Initialize API controller:
tc = TangleController(config)
is_valid = tc.validate_tangle_message(message_id=MESSAGE_IDENTIFIER,
                                      output_address=OUTPUT_ADDRESS,
                                      expected_amount=EXPECTED_AMOUNT)
if is_valid:
    logger.info("Transaction is valid.")
