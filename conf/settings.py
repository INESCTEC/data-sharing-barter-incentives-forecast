import os
import numpy as np

# Wallet Configs:
WALLET_NAME = os.environ['WALLET_NAME']
STRONG_WALLET_KEY = os.environ['STRONG_WALLET_KEY']
WALLET_STORAGE_PATH = os.environ['STORAGE_PATH']
MINIMUM_WITHDRAW_AMOUNT = int(os.environ['MINIMUM_WITHDRAW_AMOUNT'])

# REST Configs:
RESTAPI_HOST = os.environ['RESTAPI_HOST']
RESTAPI_PORT = os.environ['RESTAPI_PORT']
N_REQUEST_RETRIES = os.environ.get('N_REQUEST_RETRIES', 3)

# IOTA Configs:
IOTA_FAUCET_URL = os.environ['IOTA_FAUCET_URL']
IOTA_NODE_URL = os.environ['IOTA_NODE_URL']

# Market Configs:
MARKET_EMAIL = os.environ['MARKET_EMAIL']
MARKET_PASSWORD = os.environ['MARKET_PASSWORD']


# Market Session - First Session Configs:
class FirstSessionConfigs:
    session_number = 1
    b_min = 0.5 * 10 ** 6  # Minimum market price
    b_max = 10 * 10 ** 6  # Maximum market price
    n_price_steps = 20  # Number of price steps
    delta = 0.05  # Learning rate for price updates
    possible_p = np.linspace(start=b_min,
                             stop=b_max,
                             num=n_price_steps)
    epsilon = possible_p[1] - possible_p[0]
    weights_p = [1.] * len(possible_p)
    market_price = possible_p.mean()  # select the mean of possble prices
    market_price = (market_price // epsilon + 1) * (epsilon)
