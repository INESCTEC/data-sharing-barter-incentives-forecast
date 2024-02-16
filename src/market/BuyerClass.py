import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from .helpers.class_helpers import ValidatorClass


@dataclass()
class BuyerClass(ValidatorClass):
    market_bid_id: int = None           # Bid identifier
    resource_id: int = None             # Bid resource identifier
    user_id: int = None                 # Bid user identifier
    gain_func: str = None               # Bid gain function
    initial_bid: np.float64 = None      # Bid initial bid_price
    max_payment: np.float64 = None      # Bid max payment
    final_bid: np.float64 = None        # Bid final bid (adjusted by market)
    y: pd.DataFrame = None              # Bid resource measurements time-series
    has_to_pay: np.float64 = np.float64(0.0)  # Payment amount to bid user
    gain: np.float64 = None             # Resource id forecast estimated gain
    forecasts: pd.DataFrame = None      # Resource id market forecasts
    payment_split = {}                  # Payment division per seller
    features_list: list = field(default_factory=list)  # Suggested features

    def validate_attributes(self):
        if self.user_id is None:
            raise ValueError("BuyerClass user_id not defined.")
        if self.resource_id is None:
            raise ValueError("BuyerClass resource_id not defined.")
        if self.gain_func is None:
            raise ValueError("BuyerClass gain_func not defined.")
        if self.initial_bid is None:
            raise ValueError("BuyerClass initial_bid not defined.")
        if self.max_payment is None:
            raise ValueError("BuyerClass max_payment not defined.")
        if self.market_bid_id is None:
            raise ValueError("BuyerClass market_bid_id not defined.")
        self.validate_attr_types()
        return self

    @property
    def details(self):
        return {
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "gain_func": self.gain_func,
            "gain": self.gain,
            "initial_bid": self.initial_bid,
            "final_bid": self.final_bid,
            "max_payment": self.max_payment,
            "has_to_pay": self.has_to_pay,
            "features_list": self.features_list
        }

    @property
    def forecasts_dict(self):
        f_ = self.forecasts.copy()
        f_["resource_id"] = self.resource_id
        f_["user_id"] = self.user_id
        return f_.reset_index().to_dict(orient="records")

    def set_measurements(self, data):
        self.y = data

    def set_gain(self, gain):
        self.gain = gain

    def set_final_bid(self, price):
        self.final_bid = price

    def set_forecasts(self, forecasts):
        self.forecasts = forecasts

    def set_payment(self, price: float):
        """
        Price that the buyer will effectively have to PAY at the end of
        the market session (must be <= than initial session Market Price)

        :param price:
        :return:
        """
        self.has_to_pay = price

    def set_payment_split(self, value_dict):
        self.payment_split = value_dict
