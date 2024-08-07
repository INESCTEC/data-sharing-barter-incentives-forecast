import os
import numpy as np

from time import sleep

from conf import settings
from loguru import logger
from payment.PaymentGateway.EthereumPayment.EthereumSmartContract import (
    EthereumSmartContract,
    ethereum_provider,
    SmartContractConfig,
    TokenABI)
from payment.database.schemas.generic import (TransferList,
                                              TransactionHistorySchema,
                                              TransactionSchema,
                                              PaymentMethod,
                                              ReceiptSchema,
                                              ReceiptHistorySchema,
                                              TransferItem,
                                              TransferList)
from payment.PaymentGateway.IOTAPayment.IOTAPayment import (IOTAPaymentController,
                                                            WalletConfig)
from payment.AbstractPayment import ConversionType


def wallet_config() -> WalletConfig:

    return WalletConfig(
        node_url=[settings.IOTA_NODE_URL],
        faucet_url=settings.IOTA_FAUCET_URL,
        wallet_db_path=settings.WALLET_STORAGE_PATH,
        stronghold_snapshot_path=settings.STRONGHOLD_SNAPSHOT_PATH,
        file_dir=settings.FILE_DIR,
        wallet_backup_path=settings.WALLET_BACKUP_PATH,
        wallet_password=settings.STRONG_WALLET_KEY
    )


def smart_contract_config() -> SmartContractConfig:
    return SmartContractConfig(
        contract_address=settings.ERC20_CONTRACT_ADDRESS,
        abi=TokenABI.ETK
    )


class WalletController:
    alias = settings.WALLET_NAME

    def __init__(self, payment_type):
        print("Payment type:", payment_type)
        try:
            if payment_type == "IOTA":
                self.controller = IOTAPaymentController(
                    config=wallet_config(),
                )
            elif payment_type == "ERC20":
                config = smart_contract_config()
                provider_url = settings.WEB3_PROVIDER_URL
                if not provider_url:
                    raise ValueError("WEB3_PROVIDER_URL environment variable not set")
                w3 = ethereum_provider(url=provider_url)
                eth_private_key = os.getenv('ETH_PRIVATE_KEY', None)
                print("YOUR PRIVATE KEY IS:", eth_private_key)
                print(config)
                self.controller = EthereumSmartContract(config=config,
                                                        private_key=eth_private_key,
                                                        web3_instance=w3)
            elif payment_type == "FIAT":
                raise NotImplementedError("Fiat exchange processor not yet supported")
            else:
                raise ValueError("Unsupported exchange processor type")
        except Exception as e:
            raise e

    def transaction_to_base_units(self, transaction_amount):
        return np.float64(self.controller.unit_conversion(
            value=transaction_amount,
            unit=self.controller.TRANSACTION_UNIT,
            target_unit=self.controller.BASE_UNIT,
            conversion_type=ConversionType.TRANSACTION_TO_BASE
        ))

    def base_to_transaction_units(self, base_amount):
        return int(self.controller.unit_conversion(
            value=base_amount,
            unit=self.controller.BASE_UNIT,
            target_unit=self.controller.TRANSACTION_UNIT,
            conversion_type=ConversionType.BASE_TO_TRANSACTION
        ))

    def create_account(self):
        self.controller.initialize_payment_method()
        return self.controller.create_account(identifier=self.alias)

    def get_balance(self):
        if isinstance(self.controller, EthereumSmartContract):
            identifier = self.get_address()
        elif isinstance(self.controller, IOTAPaymentController):
            identifier = self.alias
        else:
            raise ValueError(f"Unsupported controller type {self.controller}")

        # Get balance:
        balance = self.controller.get_balance(identifier=identifier).balance

        return int(balance), self.controller.TRANSACTION_UNIT

    def get_address(self):
        return self.controller.get_account_data(identifier=self.alias).address

    def get_transaction_list(self):
        return self.controller.get_transaction_history(identifier=self.alias)

    def transfer_tokens(self, amount: int, address: str):
        if isinstance(self.controller, EthereumSmartContract):
            identifier = self.get_address()
        elif isinstance(self.controller, IOTAPaymentController):
            identifier = self.alias
        else:
            raise ValueError(f"Unsupported controller type {self.controller}")

        return self.controller.execute_transaction(from_identifier=identifier, to_identifier=address, value=amount)

    def transfer_tokens_multi_address(self, transfer_list: list):
        transactions = []
        try:
            transfers = []
            for txn in transfer_list:
                transfers.append(TransferItem(
                    from_identifier=txn["from_identifier"],
                    to_identifier=txn["to_identifier"],
                    amount=txn["amount"]
                ))
            receipts = self.controller.execute_transaction_multi_address(
                transfer_list=TransferList(transfers=transfers),
            )
            if len(receipts.transactions) == 0:
                # Failed to perform multi-output txn
                return transfer_list
            else:
                # Success performing multi-output txn.
                # Transaction ID is the same for all transactions
                transaction_id = receipts.transactions[0].receipt
                for tid in transfer_list:
                    transactions.append({
                        "user_id": tid["user_id"],
                        "from_identifier": tid["from_identifier"],
                        "to_identifier": tid["to_identifier"],
                        "amount": tid["amount"],
                        "transaction_id": transaction_id
                    })
        except NotImplementedError:
            logger.warning("Multi-output transactions are not implemented for ERC20 tokens. "
                           "Defaulting to multiple individual transactions.")
            for tid in transfer_list:
                try:
                    transaction_receipt = self.controller.execute_transaction(
                        from_identifier=tid["from_identifier"],
                        to_identifier=tid["to_identifier"],
                        value=tid["amount"]
                    )
                    transactions.append({
                        "user_id": tid["user_id"],
                        "from_identifier": transaction_receipt.from_identifier,
                        "to_identifier": transaction_receipt.to_identifier,
                        "amount": transaction_receipt.value,
                        "transaction_id": transaction_receipt.receipt
                    })
                    logger.debug(transaction_receipt)
                    logger.debug("sleeping for 15seconds.")
                    sleep(30)
                except Exception as ex:
                    logger.error(ex)
                    logger.exception(f"An error occurred processing transaction {tid}")

        return transactions

    def validate_transaction_id(self,
                                transaction_id: str,
                                to_address: str,
                                amount: int):
        # 1. Prepare transaction receipt based on predefined schemas
        # 2. Validate txn with DLT lookup
        transaction = self.controller.validate_blockchain_transactions(
            receipts=ReceiptHistorySchema(
                receipts=[
                    ReceiptSchema(
                        receipt=transaction_id,
                        to_identifier=to_address,
                        value=amount
                    )
                ]),
        )
        # 3. transaction should have a bool field = True for confirmed Txn and
        # False for unconfirmed
        if len(transaction.receipts) == 0:
            return False
        else:
            return transaction.receipts[0].confirmed
