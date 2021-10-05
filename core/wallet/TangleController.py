import iota_client

from loguru import logger

from conf import settings
from core.wallet.exception.TangleException import (
    IdNotFoundInTangle,
    IdNotSolidInTangle,
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

    def get_message(self, message_id):
        message = self.client.get_message_data(message_id)
        return message

    def get_message_output(self, message_id):
        message = self.client.get_message_data(message_id)
        # Search Message transactions "outputs" section:
        transaction = message["payload"]["transaction"][0]
        output = transaction["essence"]["outputs"]
        return output

    def validate_tangle_message(self,
                                message_id,
                                output_address,
                                expected_amount):

        logger.debug(f"Validating tangle message ID {message_id}")
        meta = self.get_message_metadata(message_id)
        message_output = self.get_message_output(message_id)

        # -- Check if is solid:
        if not meta["is_solid"]:
            raise Exception("Message is not solid yet.")
        else:
            logger.debug(f"Message ID exists and {message_id} is solid.")

        # -- Get message output transactions details:
        transactions_out = [
            x for x in message_output
            if x["signature_locked_single"]["address"] == output_address
        ]

        if len(transactions_out) > 1:
            raise Exception(f"Unexpected behaviour. Multiple transactions found "
                            f"for {output_address}")
        elif len(transactions_out) == 0:
            raise Exception(f"No transactions found to output address "
                            f"{output_address}")

        # Verify max_payment amount (IOTA) writen in Tangle transaction:
        tangle_amount = transactions_out[0]["signature_locked_single"]["amount"]

        if tangle_amount != expected_amount:
            raise Exception(f"Expected amount ({expected_amount}) differs "
                            f"from amount in Tangle ({tangle_amount})")

        logger.debug(f"Validating tangle message ID {message_id} ... Ok!")

        return True

    def validate_tangle_message_multi_output(self, message_id, transfer_list):

        logger.debug(f"Validating tangle message ID {message_id}")

        try:
            meta = self.get_message_metadata(message_id)
            message_output = self.get_message_output(message_id)
        except ValueError as ex:
            errors = {"message": ex.args[0]}
            raise IdNotFoundInTangle(message=ex.args, errors=errors)

        # -- Check if is solid:
        if not meta["is_solid"]:
            message = "Message ID is not solid yet."
            errors = {"message": message}
            raise IdNotSolidInTangle(message=message, errors=errors)
        else:
            logger.debug(f"Message ID exists and {message_id} is solid.")

        # -- Get message output transactions details:
        expected_addresses = [x["address"] for x in transfer_list]
        transactions_out = [
            x for x in message_output
            if x["signature_locked_single"]["address"] in expected_addresses
        ]

        for tt in transactions_out:
            # Verify max_payment amount (IOTA) writen in Tangle transaction:
            tangle_amount = tt["signature_locked_single"]["amount"]
            tangle_address = tt["signature_locked_single"]["address"]
            expected_amount = [x["amount"] for x in transfer_list if x["address"] == tangle_address][0]
            if tangle_amount != expected_amount:
                logger.warning(f"Expected amount ({tangle_amount}) differs from tangle_amount ({expected_amount}) for address {tangle_address}")

        logger.debug(f"Validating tangle message ID {message_id} ... Ok!")
        return True
