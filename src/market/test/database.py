
def session_price_weights_template():
    return [
        {"id": 1, "weights_p": 1.0, "market_session": 1},
        {"id": 2, "weights_p": 1.0, "market_session": 1},
        {"id": 3, "weights_p": 1.0, "market_session": 1},
        {"id": 4, "weights_p": 1.0, "market_session": 1},
        {"id": 5, "weights_p": 1.0, "market_session": 1},
        {"id": 6, "weights_p": 1.0, "market_session": 1},
        {"id": 7, "weights_p": 1.0, "market_session": 1},
        {"id": 8, "weights_p": 1.0, "market_session": 1},
        {"id": 9, "weights_p": 1.0, "market_session": 1},
        {"id": 10, "weights_p": 1.0, "market_session": 1},
        {"id": 11, "weights_p": 1.0, "market_session": 1},
        {"id": 12, "weights_p": 1.0, "market_session": 1},
        {"id": 13, "weights_p": 1.0, "market_session": 1},
        {"id": 14, "weights_p": 1.0, "market_session": 1},
        {"id": 15, "weights_p": 1.0, "market_session": 1},
        {"id": 16, "weights_p": 1.0, "market_session": 1},
        {"id": 17, "weights_p": 1.0, "market_session": 1},
        {"id": 18, "weights_p": 1.0, "market_session": 1},
        {"id": 19, "weights_p": 1.0, "market_session": 1},
        {"id": 20, "weights_p": 1.0, "market_session": 1}
    ]


def session_data_template():
    return {
            "id": 1,
            "session_number": 1,
            "session_date": "2021-12-21",
            "staged_ts": "2021-12-21T11:38:23.512205Z",
            "open_ts": "2021-12-21T11:38:27.801839Z",
            "close_ts": None,
            "launch_ts": None,
            "finish_ts": None,
            "status": "closed",
            "market_price": 5500000.0,
            "b_min": 500000,
            "b_max": 10000000,
            "n_price_steps": 20,
            "delta": 0.05
        }


def buyer_bid_template():
    return {
            "id": 1,
            "max_payment": 7000000,
            "bid_price": 5500000,
            "gain_func": "mse",
            "confirmed": True,
            "registered_at": "2021-12-21T11:41:35.532381Z",
            "has_forecasts": False,
            "user": 2,
            "resource": 1,
            "market_session": 1
        }


def user_resources_template():
    return {
            "id": 1,
            "name": "resource-4",
            "type": "measurements",
            "to_forecast": True,
            "registered_at": "2021-12-20T19:06:33.773648Z",
            "user": 2
        }