from psycopg2.errors import UniqueViolation, ForeignKeyViolation
import pandas as pd
import datetime as dt

from loguru import logger
from src.database.PostgresDB import PostgresDB


# #############################################################################
# Get Measurements Data for Session Resources:
# #############################################################################


def get_measurements_data_mock(users_resources, market_launch_time):
    resources_id_list = [x["id"] for x in users_resources]
    # todo: Right now working with mock data - replace by DB queries:
    from ..util.mock import MeasurementsGenerator
    logger.info("[MOCK] Querying measurements for resource list ...")
    # Create fictitious measurements data
    mg = MeasurementsGenerator()
    measurements = {}
    for resource_id in sorted(resources_id_list):
        measurements[resource_id] = mg.generate_mock_data_sin(
            start_date=market_launch_time - pd.DateOffset(months=12),
            end_date=market_launch_time,
        ).set_index("datetime")
    logger.info("[MOCK] Querying measurements for resource list ... Ok!")
    return measurements


def get_measurements_data(users_resources, market_launch_time):
    resources_id_list = [x["id"] for x in users_resources]
    db = PostgresDB.get_db_instance(config_name="default")
    logger.info("Querying measurements for resource list ...")
    measurements = {}
    for resource_id in sorted(resources_id_list):
        logger.debug(f"Querying for resource ID {resource_id} ...")
        query = f"select datetime, value " \
                f"from raw_data " \
                f"where resource_id={resource_id} " \
                f"and datetime <= '{market_launch_time}' " \
                f"order by datetime asc;"
        data = db.read_query_pandas(query)
        if data.empty:
            logger.warning(f"No historical data for resource ID {resource_id}")
            measurements[resource_id] = data
        else:
            # todo: improve data processing pipeline
            data = data.set_index("datetime")
            data = data.resample("h").mean().dropna()
            measurements[resource_id] = data
        logger.debug(f"Querying for resource ID {resource_id} ... Ok!")
    logger.info("Querying measurements for resource list ... Ok!")
    return measurements


# #############################################################################
# Upload market forecasts for session resources:
# #############################################################################

def upload_forecasts(market_session_id,
                     user_id,
                     request,
                     resource_id,
                     forecasts,
                     table_name):
    # Create datetime col:
    forecasts.reset_index(drop=False, inplace=True)
    # Create other cols:
    forecasts["request"] = request
    forecasts["market_session_id"] = market_session_id
    forecasts["user_id"] = user_id
    forecasts["registered_at"] = dt.datetime.utcnow()
    forecasts["units"] = "kw"  # Todo: Assure this is dynamic:
    forecasts["resource_id"] = resource_id

    forecasts = forecasts[["datetime", "request", "value", "units",
                           "registered_at", "market_session_id",
                           "resource_id", "user_id"]]

    # Insert data in DB:
    try:
        db = PostgresDB.get_db_instance(config_name="default")
        logger.debug(f"Forecast shape: {forecasts.shape}")
        logger.debug(f"Inserting agent {user_id} forecasts ...")
        db.insert_dataframe(df=forecasts, table=table_name)
        logger.debug(f"Inserting agent {user_id} forecasts ... Ok!")
        return True
    except (UniqueViolation, ForeignKeyViolation) as ex:
        msg = f"Failed to insert agent {user_id} forecasts"
        logger.error(f"{msg} - {ex}")
    except Exception:
        msg = f"Unexpected error while inserting agent {user_id} forecasts"
        logger.exception(msg)
        return False


def update_bid_has_forecast(user_id, bid_id, table_name):
    try:
        db = PostgresDB.get_db_instance(config_name="default")
        logger.debug(f"Updating {user_id} - bid {bid_id} "
                     f"'has_forecast' field ...")
        query = f"UPDATE {table_name} " \
                f"SET has_forecasts = true " \
                f"WHERE id = {bid_id};"
        db.execute_query(query=query)
        logger.debug(f"Updating {user_id} - bid {bid_id} "
                     f"'has_forecast' field ... Ok!")
        return True
    except Exception:
        logger.exception(f"Failed update {user_id} - bid {bid_id} "
                         f"'has_forecast' field.")
        return False
