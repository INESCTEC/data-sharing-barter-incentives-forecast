import numpy as np
import pandas as pd

from dataclasses import dataclass
from .helpers.class_helpers import ValidatorClass


@dataclass
class SellerClass(ValidatorClass):
    user_id: int = None                 # Resource User ID
    resource_id: int = None             # Resource ID
    y: pd.DataFrame = None              # Resource measurements time-series
    has_to_receive: np.float64 = np.float64(0.0)  # Resource revenue

    def validate_attributes(self):
        if self.user_id is None:
            raise ValueError("BuyerClass user_id not defined.")
        if self.resource_id is None:
            raise ValueError("BuyerClass resource_id not defined.")
        self.validate_attr_types()

    @property
    def details(self):
        return {
            "user_id": self.user_id,
            "resource_id": self.resource_id,
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
