import iota_client

from loguru import logger

from conf import settings


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

        if len(transactions_out) == 0:
            raise Exception(f"No transactions found to output address "
                            f"{output_address}")

        # Verify max_payment amount (IOTA) writen in Tangle transaction:
        tangle_amount = transactions_out[0]["signature_locked_single"]["amount"]

        if tangle_amount != expected_amount:
            raise Exception(f"Expected amount ({expected_amount}) differs "
                            f"from amount in Tangle ({tangle_amount})")

        logger.debug(f"Validating tangle message ID {message_id} ... Ok!")

        return True
