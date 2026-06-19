import numpy as np
import datetime as dt

from dataclasses import dataclass
from .helpers.class_helpers import ValidatorClass


@dataclass()
class SessionClass(ValidatorClass):
    b_min: np.float64 = None                # Minimum market price
    b_max: np.float64 = None                # Maximum market price
    market_price: np.float64 = None         # Session Market Price
    next_market_price: np.float64 = None    # Next session market price
    status: str = None                      # Session status
    session_id: int = None                  # Session ID (unique)
    session_number: int = None              # Session ID (unique per date)
    session_date: dt.date = None        # Session Date (%Y-%m-%d)
    launch_ts: dt.datetime = None           # Session start timestamp (launch)
    finish_ts: dt.datetime = None           # Session end timestamp (finish)
    prev_weights_p: np.ndarray = None       # Previous session price weights
    next_weights_p: np.ndarray = None       # Next session price weights
    buyers_results: dict = None             # Session results by buyer
    sellers_results: dict = None            # Session results by seller
    buyers_forecasts: dict = None           # Session forecasts by buyer
    n_price_steps: int = None               # Number of price steps
    delta: np.float64 = None                # Learning rate for price updates
    possible_p: np.ndarray = None           # Array of possible price refs
    epsilon: np.float64 = None              # Interval in possible_p array
    status_list = ["open", "closed", "running", "finished"]
    total_market_fee = 0
    market_fee_per_resource = {}

    def set_initial_conditions(self):
        # possible prices:
        self.possible_p = np.linspace(start=self.b_min,
                                      stop=self.b_max,
                                      num=self.n_price_steps)
        if len(self.prev_weights_p) != len(self.possible_p):
            raise ValueError("SessionClass prev_weights_p has wrong dim.")
        self.epsilon = self.possible_p[1] - self.possible_p[0]
        self.buyers_results = {}
        self.sellers_results = {}
        self.buyers_forecasts = {}

    def validate_attributes(self):
        if self.session_id is None:
            raise ValueError("SessionClass identifier not defined.")
        elif self.market_price is None:
            raise ValueError("SessionClass market_price not defined.")
        elif self.prev_weights_p is None:
            raise ValueError("SessionClass prev_weights_p not defined.")
        elif (self.b_min is None) or (self.b_max is None):
            raise ValueError("SessionClass b_min or b_max not defined.")
        elif self.n_price_steps is None:
            raise ValueError("SessionClass n_price_steps not defined.")
        elif self.delta is None:
            raise ValueError("SessionClass price_delta not defined.")

        if isinstance(self.session_date, str):
            self.session_date = dt.datetime.strptime(self.session_date, "%Y-%m-%d").date()  # noqa

        # Readjust type of session delta after init:
        self.delta = np.float64(self.delta)

        # Validate parameters types:
        self.validate_attr_types()

    @property
    def details(self):
        return {
            "market_price": self.market_price,
            "next_market_price": self.next_market_price,
            "session_id": self.session_id,
            "session_number": self.session_number,
            "session_date": self.session_date.strftime("%Y-%m-%d %H:%M:%S"),
            "status": self.status,
            "launch_ts": self.launch_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "next_weights_p": str(self.next_weights_p),
            "prev_weights_p": str(self.prev_weights_p),
            "total_market_fee": self.total_market_fee,
            "market_fee_per_resource": self.market_fee_per_resource,
        }

    @property
    def buyer_payment_per_user(self):
        # Get list of payments per user_id
        users_ = [(x['user_id'], x["has_to_pay"])
                  for x in self.buyers_results.values()]

        # Create a dictionary to store the sum of second elements
        # for each first element
        result = {}
        for key, value in users_:
            result[key] = result.get(key, 0) + value

        output = [{"user": key, "has_to_pay": value}
                  for key, value in result.items()]

        return output

    @property
    def seller_revenue_per_user(self):
        # Get list of payments per user_id
        users_ = [(x['user_id'], x["has_to_receive"], x["shapley_value"])
                  for x in self.sellers_results.values()]

        # Create a dictionary to store the sum of second elements
        # for each first element
        result = {}
        for key, has_to_receive, sv in users_:
            result[key] = (result.get(key, 0) + has_to_receive, result.get(key, 0) + sv)

        output = [{"user": key, "has_to_receive": value[0], "shapley_value": value[1]}
                  for key, value in result.items()]

        return output

    def set_previous_price_weights(self, weights_p):
        self.prev_weights_p = weights_p

    def set_next_price_weights(self, weights_p):
        self.next_weights_p = weights_p

    def set_next_market_price(self, price):
        self.next_market_price = np.float64(price)

    def set_buyer_result(self, buyer_cls):
        self.buyers_results[buyer_cls.resource_id] = buyer_cls.details

    def set_seller_result(self, seller_cls):
        self.sellers_results[seller_cls.resource_id] = seller_cls.details

    def set_buyer_forecasts(self, buyer_cls):
        self.buyers_forecasts[buyer_cls.resource_id] = buyer_cls.forecasts_dict

    def add_market_fee(self, resource_id, value):
        self.market_fee_per_resource[resource_id] = value
        self.total_market_fee += value

    def start_session(self):
        self.status = "running"

    def end_session(self):
        self.status = "finished"
        self.finish_ts = dt.datetime.utcnow()
