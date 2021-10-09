import numpy as np
import pandas as pd

from dataclasses import dataclass
from .helpers.class_helpers import ValidatorClass


@dataclass
class SellerClass(ValidatorClass):
    identifier: int = None              # Seller identifier
    # bid: np.float64 = None            # Seller bid price (not in use)
    y: pd.DataFrame = None              # Seller measurements time-series
    has_to_receive: np.float64 = np.float64(0.0)  # Seller revenue

    def validate_attributes(self):
        if self.identifier is None:
            raise ValueError("BuyerClass identifier not defined.")
        self.validate_attr_types()

    @property
    def details(self):
        return {
            "identifier": self.identifier,
            "has_to_receive": self.has_to_receive,
        }

    def set_measurements(self, data):
        self.y = data

    def increment_revenue(self, price: float):
        """
        Price that the buyer will effectively have to RECEIVE at the end of
        the market session

        :param price:
        :return:
        """
        self.has_to_receive += price
