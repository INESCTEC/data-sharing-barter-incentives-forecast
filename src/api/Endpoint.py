from dataclasses import dataclass
from collections import namedtuple

fields = ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'uri')
endpoint = namedtuple('endpoint', fields, defaults=(None,) * len(fields))

# HTTP methods
http_methods = "GET", "POST", "PUT", "DELETE", "PATCH",

# Authentication
login = endpoint(*http_methods, "/api/token/login")
register = endpoint(*http_methods, "/api/user/register")

# User & Role
user_list = endpoint(*http_methods, "/api/user/list")
user_resources = endpoint(*http_methods, "/api/user/resource")
wallet_address = endpoint(*http_methods, "/api/user/wallet-address")
market_wallet_address = endpoint(*http_methods, "/api/market/wallet-address")

# Market endpoints
market_session = endpoint(*http_methods, "/api/market/session")
market_balance = endpoint(*http_methods, "/api/market/balance")
market_session_balance = endpoint(*http_methods, "/api/market/session-balance")
market_session_fee = endpoint(*http_methods, "/api/market/session-fee")
market_bid = endpoint(*http_methods, "/api/market/bid")
market_validate_bids = endpoint(*http_methods, "/api/market/validate/bid-payment")
market_transfer_out = endpoint(*http_methods, "/api/market/transfer-out")
market_payment = endpoint(*http_methods, "/api/market/payment")
market_price_weight = endpoint(*http_methods, "/api/market/price-weight")


@dataclass(frozen=True)
class Endpoint:
    http_method: str
    uri: str
