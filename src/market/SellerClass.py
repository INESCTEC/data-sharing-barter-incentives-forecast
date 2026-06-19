import numpy as np
import pandas as pd

from typing import Union
from dataclasses import dataclass
from .helpers.class_helpers import ValidatorClass


@dataclass
class SellerClass(ValidatorClass):
    user_id: Union[int, str] = None                 # Resource User ID
    resource_id: Union[int, str] = None             # Resource ID
    resource_type: str = None           # Type (measurements or features)
    y: pd.DataFrame = None              # Resource measurements time-series
    has_to_receive: np.float64 = np.float64(0.0)  # Resource revenue
    shapley_value: np.float64 = np.float64(0.0)  # shapley_value

    def validate_attributes(self):
        if self.user_id is None:
            raise ValueError("SellerClass user_id not defined.")
        if self.resource_id is None:
            raise ValueError("SellerClass resource_id not defined.")
        if self.resource_type not in ["features", "measurements"]:
            raise ValueError("SellerClass invalid resource_type.")
        self.validate_attr_types()
        return self

    @property
    def details(self):
        return {
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "has_to_receive": self.has_to_receive,
            "shapley_value": self.shapley_value
        }

    def set_data(self, data):
        self.y = data

    def increment_revenue(self, price: float):
        """
        Price that the buyer will effectively have to RECEIVE at the end of
        the market session

        :param price:
        :return:
        """
        self.has_to_receive += price

    def increment_shapley_value(self, value_dict):
        self.shapley_value += value_dict
