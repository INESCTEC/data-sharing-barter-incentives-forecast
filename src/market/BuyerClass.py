import numpy as np
import pandas as pd

from dataclasses import dataclass
from .helpers.class_helpers import ValidatorClass


@dataclass()
class BuyerClass(ValidatorClass):
    market_bid_id: int = None           # Bid identifier
    identifier: int = None              # Buyer identifier
    gain_func: str = None               # Buyer gain function
    initial_bid: np.float64 = None      # Buyer initial bid
    max_payment: np.float64 = None      # Buyer max payment
    final_bid: np.float64 = None        # Buyer final bid (adjusted by market)
    y: pd.DataFrame = None              # Buyer measurements time-series
    has_to_pay: np.float64 = np.float64(0.0)    # Buyer payment amount
    gain: np.float64 = None             # Buyer forecast gain
    forecasts: pd.DataFrame = None      # Buyer forecasts
    payment_split = {}                  # Payment division per seller

    def validate_attributes(self):
        if self.identifier is None:
            raise ValueError("BuyerClass identifier not defined.")
        if self.gain_func is None:
            raise ValueError("BuyerClass gain_func not defined.")
        if self.initial_bid is None:
            raise ValueError("BuyerClass initial_bid not defined.")
        if self.max_payment is None:
            raise ValueError("BuyerClass max_payment not defined.")
        if self.market_bid_id is None:
            raise ValueError("BuyerClass market_bid_id not defined.")
        self.validate_attr_types()

    @property
    def details(self):
        return {
            "identifier": self.identifier,
            "gain_func": self.gain_func,
            "gain": self.gain,
            "initial_bid": self.initial_bid,
            "final_bid": self.final_bid,
            "max_payment": self.max_payment,
            "has_to_pay": self.has_to_pay
        }

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
