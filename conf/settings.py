import os
import numpy as np

# Wallet Configs:
WALLET_NAME = os.getenv('WALLET_NAME', 'wallet')
STRONG_WALLET_KEY = os.getenv('STRONG_WALLET_KEY', '123456')
WALLET_STORAGE_PATH = os.path.join(os.getenv('STORAGE_PATH', 'files'), 'payment.db')
MINIMUM_WITHDRAW_AMOUNT = int(os.getenv('MINIMUM_WITHDRAW_AMOUNT', 1000000))
STRONGHOLD_SNAPSHOT_PATH = os.path.join(os.getenv('WALLET_STORAGE_PATH', 'files'), 'stronghold.snapshot')
FILE_DIR = os.getenv('FILE_DIR', 'files')
WALLET_BACKUP_PATH = os.path.join(os.getenv('WALLET_BACKUP_PATH', 'files'), 'backup.db')


# REST Configs:
RESTAPI_HOST = os.environ['RESTAPI_HOST']
RESTAPI_PORT = os.environ['RESTAPI_PORT']
N_REQUEST_RETRIES = os.environ.get('N_REQUEST_RETRIES', 3)

# IOTA Configs:
IOTA_FAUCET_URL = os.getenv('IOTA_FAUCET_URL', 'https://faucet.testnet.shimmer.network')
IOTA_NODE_URL = os.getenv('IOTA_NODE_URL', 'https://api.testnet.shimmer.network')


# Market Configs:
RUN_REAL_MARKET = (os.getenv('RUN_REAL_MARKET', 'false').lower() == 'true')
MARKET_EMAIL = os.environ['MARKET_EMAIL']
MARKET_PASSWORD = os.environ['MARKET_PASSWORD']
N_JOBS = int(os.environ["N_JOBS"])

# Database configs:
DATABASES = {
    'default': {
        'NAME': os.environ.get("POSTGRES_NAME", default=''),
        'USER': os.environ.get("POSTGRES_USER", default=''),
        'PASSWORD': os.environ.get("POSTGRES_PASSWORD", default=''),
        'HOST': os.environ.get("POSTGRES_HOST", default=''),
        'PORT': int(os.environ.get("POSTGRES_PORT", default=5432)),
    }
}


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
