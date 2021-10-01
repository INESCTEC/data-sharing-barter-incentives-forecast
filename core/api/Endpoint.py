from dataclasses import dataclass
from collections import namedtuple

fields = ('GET', 'POST', 'PUT', 'DELETE', 'uri')
endpoint = namedtuple('endpoint', fields, defaults=(None,) * len(fields))

# HTTP methods
http_methods = "GET", "POST", "PUT", "DELETE",

# Authentication
login = endpoint(*http_methods, "/api/token/login")
register = endpoint(*http_methods, "/api/user/register")

# User & Role
user_list = endpoint(*http_methods, "/api/user/list")
user_role = endpoint(*http_methods, "/api/user/role")
role_by_user = endpoint(*http_methods, "/api/user/role-by-user/")
wallet_address = endpoint(*http_methods, "/api/user/wallet-address")

# Market endpoints
market_session = endpoint(*http_methods, "/api/market/session")
market_balance = endpoint(*http_methods, "/api/market/balance")
market_session_balance = endpoint(*http_methods, "/api/market/session-balance")
market_bid = endpoint(*http_methods, "/api/market/bid")
market_validate_bids = endpoint(*http_methods, "/api/market/validate/bid-payment")
market_transfer_out = endpoint(*http_methods, "/api/market/transfer-out")
market_payment = endpoint(*http_methods, "/api/market/payment")
market_price_weight = endpoint(*http_methods, "/api/market/price-weight")


# -- Wallet endpoints:
market_wallet_address = endpoint(*http_methods, "/api/market/wallet-address")
wallet_create = endpoint(*http_methods, "/api/wallet/create")
wallet_account = endpoint(*http_methods, "/api/wallet/account")
wallet_list = endpoint(*http_methods, '/api/wallet/list')
wallet_balance = endpoint(*http_methods, "/api/wallet/balance")
wallet_withdraw = endpoint(*http_methods, "/api/wallet/withdraw")

wallet_withdraw_approval = endpoint(*http_methods,
                                    "/api/wallet/admin/withdraw")


@dataclass(frozen=True)
class Endpoint:
    http_method: str
    uri: str
