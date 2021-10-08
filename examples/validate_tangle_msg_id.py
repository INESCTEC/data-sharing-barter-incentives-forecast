import os

from loguru import logger
from dotenv import load_dotenv

__ENV__ = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(__ENV__)

from core.wallet.TangleController import TangleController


MESSAGE_ID = "e2dc55d91b6d659be777b962aabee8d3686a68c444de9992d223ad513c7d517b"
AMOUNT = 10000000
OUTPUT_ADDRESS = "atoi1qr4k93up3mpzhuqfyn8sws5890tsx4e0xald3dzef7ggp6pd6eu9xrqdd8q"

tc = TangleController()
valid = tc.validate_tangle_message(
    message_id=MESSAGE_ID,
    expected_amount=AMOUNT,
    output_address=OUTPUT_ADDRESS
)
logger.info(f"Is valid: {valid}")
