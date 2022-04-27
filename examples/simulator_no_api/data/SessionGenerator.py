import os
import random
import numpy as np
import pandas as pd
import datetime as dt


class SessionGenerator:
    session_number = 1
    b_min = 0.5 * 10 ** 6  # Minimum market price
    b_max = 10 * 10 ** 6  # Maximum market price
    n_price_steps = 20  # Number of price steps
    delta = 0.05  # Learning rate for price updates
    possible_p = np.linspace(start=b_min,
                             stop=b_max,
                             num=n_price_steps)
    epsilon = np.float64(possible_p[1] - possible_p[0])

    def __init__(self):
        self.price_weights = None
        self.market_price = None
        self.session_id = None
        self.launch_time = None
        self.session_date = None
        self.status = None

    def create_session(self, session_id, launch_time):
        self.session_id = session_id
        self.launch_time = launch_time
        self.status = "running"
        if self.market_price is None:
            # possible prices mean
            self.market_price = self.possible_p.mean()
            self.market_price = (self.market_price // self.epsilon + 1)
            self.market_price *= self.epsilon
            # initial weights in price
            self.price_weights = np.repeat(1.0, len(self.possible_p))

    @property
    def session_data(self):
        return {'b_max': self.b_max,
                'b_min': self.b_min,
                'close_ts': None,
                'delta': self.delta,
                'finish_ts': None,
                'id': self.session_id,
                'launch_ts': None,
                'market_price': self.market_price,
                'n_price_steps': self.n_price_steps,
                'open_ts': None,
                'session_date': self.launch_time.date(),
                'session_number': self.session_id,
                'staged_ts': None,
                'status': self.status}

    def set_market_price(self, market_price):
        self.market_price = market_price

    def set_price_weights(self, price_weights):
        self.price_weights = price_weights
