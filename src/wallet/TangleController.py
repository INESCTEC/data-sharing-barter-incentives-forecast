import iota_client

from loguru import logger

from conf import settings
from src.wallet.exception.TangleException import (
    IdNotFoundInTangle,
    IdNotConfirmedInTangle,
)


class TangleController:
    node_sync_disabled = True

    def __init__(self):
        node_url = settings.IOTA_NODE_URL  # this can be a list of nodes?
        self.nodes = [[node_url, ]]
        self.client_options = {
            "nodes_name_password": self.nodes,
            "node_sync_disabled": self.node_sync_disabled
        }
        self.client = iota_client.Client(**self.client_options)

    def get_message_metadata(self, message_id):
        metadata = self.client.get_message_metadata(message_id)
        return metadata

    def get_message_meta_and_data(self, message_id):
        # Search message metadata:
        metadata = self.client.get_message_metadata(message_id)
        # Search message transactions "outputs" section:
        data = self.client.get_message_data(message_id)
        return metadata, data

    @staticmethod
    def __is_msg_confirmed(message_metadata):
        solid = message_metadata["is_solid"]
        try:
            included_in_ledger = message_metadata["ledger_inclusion_state"] \
                                     .get("state", "not_included") \
                                     .lower() == "included"
        except AttributeError:
            included_in_ledger = False

        if solid and included_in_ledger:
            logger.debug("Message is solid and included in ledger milestone.")
            return True
        elif solid and (not included_in_ledger):
            logger.warning("Message is solid but is is not yet included in a "
                           "ledger milestone.")
            return False
        elif (not solid) and included_in_ledger:
            logger.warning("Message is included in ledger milestone but is "
                           "not solid.")
            return False
        else:
            logger.warning("Message not solid nor included in ledger "
                           "milestone.")
            return False

    def check_if_reattached(self, message_meta, message_data):
        """
        Check if current message was already reattached. If so, validate
        based on that reattached message body.

        :return:
        """

        try:
            # Get output transaction ID (based on message payload):
            logger.debug("Searching for reattached message ...")
            tx_id = self.client.get_transaction_id(message_data["payload"])
            logger.debug(f"Output transaction id: {tx_id}")
            logger.debug("Searching for reattached message ... Ok!")
        except Exception:
            logger.exception("Failed to find reattached message!")
            return False, message_meta, message_data

        try:
            # Search for message assigned to new output transaction ID:
            logger.debug("Getting data and meta for reattached msg ...")
            reatached_data = self.client.get_included_message(tx_id)
            reattached_id = reatached_data["message_id"]
            logger.debug(f"ID of reattached message: {reattached_id}")
            reattached_meta = self.client.get_message_metadata(reattached_id)
            logger.debug("Getting data and meta for reattached msg ... Ok!")
        except Exception:
            logger.exception("Failed to get data and meta for reattached msg.")
            return False, message_meta, message_data

        # Check if msg is confirmed:
        confirmed = self.__is_msg_confirmed(message_metadata=reattached_meta)
        return confirmed, reattached_meta, reatached_data

    def validate_message(self, output_type, message_id, **kwargs):
        logger.debug(f"Validating tangle message ID {message_id}")

        try:
            # -- Get Message metadata details:
            meta, data = self.get_message_meta_and_data(message_id)
        except ValueError as ex:
            errors = {"message": ex.args[0]}
            raise IdNotFoundInTangle(message=ex.args, errors=errors)

        # Check if tangle message ID is solid & included in ledger milestone
        confirmed = self.__is_msg_confirmed(message_metadata=meta)
        if not confirmed:
            # Check if there is a reattached msg:
            confirmed, meta, data = self.check_if_reattached(
                message_meta=meta,
                message_data=data,
            )
            # todo: if message is reattached, we should inform DB about it
            if not confirmed:
                # raise error if message is not reattached / confirmed:
                message = "Message ID is not confirmed in tangle yet and " \
                          "no reattachment's were found. Try again later."
                errors = {"message": message}
                raise IdNotConfirmedInTangle(message, errors)

        if output_type == "single":
            confirmed = self.validate_single_output_message(
                message_data=data,
                **kwargs)
            logger.debug(f"Validating tangle message ID {message_id} ... Ok!")
            return confirmed
        elif output_type == "multiple":
            confirmed = self.validate_multi_output_message(
                message_data=data,
                **kwargs)
            logger.debug(f"Validating tangle message ID {message_id} ... Ok!")
            return confirmed
        else:
            raise AttributeError("output_type must be 'single' or 'multiple'")

    @staticmethod
    def validate_single_output_message(message_data: dict,
                                       output_address: str,
                                       expected_amount: int):

        # Get Message transactions "outputs" section:
        transaction = message_data["payload"]["transaction"][0]
        message_output = transaction["essence"]["outputs"]

        # Filter transactions to desired output address:
        transactions_out = [
            x for x in message_output
            if x["signature_locked_single"]["address"] == output_address
        ]

        # Check if there is only 1 transaction (expected behaviour):
        if len(transactions_out) > 1:
            raise Exception(f"Unexpected behaviour. "
                            f"Multiple transactions found "
                            f"for {output_address}")
        elif len(transactions_out) == 0:
            raise Exception(f"No transactions found to output address "
                            f"{output_address}")

        # Verify max_payment amount (IOTA) writen in Tangle transaction:
        tangle_amount = transactions_out[0]["signature_locked_single"]["amount"]

        # Check if amount in tangle msg == amount registered in bid
        if tangle_amount != expected_amount:
            raise Exception(f"Expected amount ({expected_amount}) differs "
                            f"from amount in Tangle ({tangle_amount})")
        else:
            logger.debug(f"Amount in tangle ({tangle_amount}) matches "
                         f"expected amount ({expected_amount})")

        return True

    @staticmethod
    def validate_multi_output_message(message_data: dict,
                                      transfer_list: list):

        # Get Message transactions "outputs" section:
        transaction = message_data["payload"]["transaction"][0]
        message_output = transaction["essence"]["outputs"]

        # -- Get message output transactions for expected output address:
        expected_addresses = [x["address"] for x in transfer_list]
        transactions_out = [
            x for x in message_output
            if x["signature_locked_single"]["address"] in expected_addresses
        ]

        # Verify if each expected transaction is in tangle,
        # with correct amount, to the correct output address
        for tt in transactions_out:
            # Verify max_payment amount (IOTA) writen in Tangle transaction:
            tangle_amount = tt["signature_locked_single"]["amount"]
            tangle_address = tt["signature_locked_single"]["address"]
            expected_amount = [x["amount"] for x in transfer_list
                               if x["address"] == tangle_address][0]
            if tangle_amount != expected_amount:
                logger.warning(f"Expected amount ({tangle_amount}) differs "
                               f"from tangle_amount ({expected_amount}) "
                               f"for address {tangle_address}")

        return True
