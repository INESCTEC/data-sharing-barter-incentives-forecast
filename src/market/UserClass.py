import numpy as np

from typing import Union
from dataclasses import dataclass, field

from .helpers.class_helpers import ValidatorClass


@dataclass()
class UserClass(ValidatorClass):
    user_id: Union[int, str] = None
    user_features_list: list = field(default_factory=list)
    total_payment: np.float64 = np.float64(0.0)
    total_revenue: np.float64 = np.float64(0.0)

    @property
    def details(self):
        return {
            "user_id": self.user_id,
            "user_features_list": self.user_features_list,
            "total_payment": self.total_payment,
            "total_revenue": self.total_revenue,
        }

    def sum_payment(self, value):
        self.total_payment += value

    def sum_revenue(self, value):
        self.total_revenue += value
