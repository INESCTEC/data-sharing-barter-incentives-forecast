import os

import pandas as pd
import pytest

from .common import (
    init_market_class,
    create_market_session_data,
    create_session_price_weights
)

__TEST_DATA_DIR__ = os.path.join(os.path.dirname(__file__), "files")


@pytest.fixture
def market_launch_time():
    lt = '2020-05-01 10:00:00'
    return pd.to_datetime(lt).tz_localize("UTC").to_pydatetime()


@pytest.fixture
def price_weights():
    return create_session_price_weights()


@pytest.fixture
def session_data():
    return create_market_session_data(convert_to_miota=True)


@pytest.fixture
def session_class(market_launch_time, session_data, price_weights):
    return init_market_class(
        launch_time=market_launch_time,
        session_data=session_data,
        price_weights_data=price_weights
    )

