# flake8: noqa

import os
import numpy as np

from loguru import logger
from dataclasses import dataclass

# Pathing:
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Logs Configs:
LOGS_DIR = os.path.join(BASE_PATH, "files", "logs")

# -- Initialize Logger:
logs_kw = dict(
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    rotation="1 week",
    compression="zip",
    backtrace=True,
)
logger.add(os.path.join(LOGS_DIR, "info_log.log"), level='INFO', **logs_kw)
logger.add(os.path.join(LOGS_DIR, "debug_log.log"), level='DEBUG', **logs_kw)

# Wallet Configs:
WALLET_NAME = os.getenv('WALLET_NAME', 'wallet')
STRONG_WALLET_KEY = os.getenv('STRONG_WALLET_KEY', '123456')
WALLET_STORAGE_PATH = os.path.join(os.getenv('STORAGE_PATH', 'files'), 'payment.db')
MINIMUM_WITHDRAW_AMOUNT = int(os.getenv('MINIMUM_WITHDRAW_AMOUNT', 1000000))
STRONGHOLD_SNAPSHOT_PATH = os.path.join(os.getenv('WALLET_STORAGE_PATH', 'files'), 'stronghold.snapshot')
FILE_DIR = os.getenv('FILE_DIR', 'files')
WALLET_BACKUP_PATH = os.path.join(os.getenv('WALLET_BACKUP_PATH', 'files'), 'backup.db')

# REST Configs:
RESTAPI_HOST = os.environ.get('RESTAPI_HOST', "")
RESTAPI_PORT = os.environ.get('RESTAPI_PORT', "")
N_REQUEST_RETRIES = os.environ.get('N_REQUEST_RETRIES', 3)

# IOTA Configs:
IOTA_FAUCET_URL = os.getenv('IOTA_FAUCET_URL', 'https://faucet.testnet.shimmer.network')
IOTA_NODE_URL = os.getenv('IOTA_NODE_URL', 'https://api.testnet.shimmer.network')

# Market Configs:
RUN_REAL_MARKET = (os.getenv('RUN_REAL_MARKET', 'false').lower() == 'true')
MARKET_EMAIL = os.environ.get('MARKET_EMAIL', "")
MARKET_PASSWORD = os.environ.get('MARKET_PASSWORD', "")
N_JOBS = int(os.environ.get("N_JOBS", 1))
MARKET_FORECAST_HORIZON = int(os.environ.get("MARKET_FORECAST_HORIZON", 24))

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


@dataclass(frozen=True)
class FeaturePreprocess:
    feature_selection = dict(
        seed=42,

        # General Setting for Feature Selection
        method_fs='Spearman',  # 'mRMR', 'Pearson', 'Spearman', 'MI', 'Partial-Pearson', 'Partial-Spearman',   # noqa

        # selection method
        type_selection='thresh',
        percentile=99,
        threshold=0.00001,

        # statistical-based filters
        significance_level=0.1,

        # mutual information params
        nr_neighbors=5,

        # results
        path_to_save_fs='./results_feature_selection/',
        dir_fs='feature_selection',
        filename_scores='scores.csv',
        filename_fs='feature_selected.csv',
        format='json',
    )


@dataclass(frozen=True)
class AutocorrelationAnalysis:
    # Auto-correlation analysis configs:
    acf_kwargs = {
        "nlags": 504,
        "threshold": 0.3,
        "select_top": 2
    }
