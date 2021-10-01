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

from dotenv import load_dotenv

__ENV_PATH__ = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(__ENV_PATH__)

from conf import settings
from core.api import Controller


def initialize_market():
    # Initialize API controller:
    controller = Controller()

    # Login into market REST:
    controller.login(email=settings.MARKET_EMAIL,
                     password=settings.MARKET_PASSWORD)

    # Create market roles:
    controller.create_market_role(role="buyer")
    controller.create_market_role(role="seller")


if __name__ == '__main__':
    initialize_market()
