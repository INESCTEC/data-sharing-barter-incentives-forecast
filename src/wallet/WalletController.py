import os
import iota_wallet as iw

from loguru import logger

from conf import settings
from .helpers.wallet_helper import transaction_report
from .exception.WalletException import InsufficientFundsException


class WalletController:

    local_pow = True
    alias = settings.WALLET_NAME

    def __init__(self):
        self.node_url = settings.IOTA_NODE_URL
        stronghold_pw = settings.STRONG_WALLET_KEY
        self.wallet_location = os.path.join(settings.WALLET_STORAGE_PATH,
                                            "wallet-db")
        self.client_options = {
            "nodes": [{"url": self.node_url,
                       "auth": None,
                       "disabled": False}],
            "local_pow": self.local_pow
        }
        self.account_manager = iw.AccountManager(
            storage_path=self.wallet_location,
            allow_create_multiple_empty_accounts=True
        )
        self.account_manager.set_stronghold_password(stronghold_pw)

    def create_wallet(self, store_mnemonic=False):
        self.account_manager.store_mnemonic('Stronghold')
        mnemonic = self.account_manager.generate_mnemonic()
        print(f"Your wallet mnemonic:\n{mnemonic}")
        if store_mnemonic:
            file_path = os.path.join(self.wallet_location, "mnemonic.txt")
            with open(file_path, "w") as f:
                f.write(mnemonic)
            print(f"Your mnemonic was stored in {self.wallet_location}.")
        print("A new wallet DB was created.")

    def create_account(self):
        account_initializer = self.account_manager.create_account(
            self.client_options
        )
        account_initializer.alias(self.alias)
        account = account_initializer.initialise()
        print(f'Account created: {account.alias()}')

    def get_balance(self):
        account = self.account_manager.get_account(self.alias)
        account.sync().execute()
        return account.balance()

    def get_address(self):
        """
        :return: address object
        """
        for _ in range(5):
            try:
                account = self.account_manager.get_account(self.alias)
                account.sync().execute()
                return account.generate_address()
            except TimeoutError as e:
                raise e
            except ValueError as e:
                connection_refused = 'Connection refused'
                timeout_err = 'operation timeout'
                if connection_refused in str(e):
                    raise ConnectionRefusedError("Unable to communicate with node!")
                if timeout_err in str(e):
                    import time
                    time.sleep(5)
                    continue
        raise Exception("It was not possible to generate the wallet address")

    def get_transaction_list(self):
        account = self.account_manager.get_account(self.alias)
        account.sync().execute()
        data = transaction_report(account.list_messages())
        return data

    def transfer_tokens(self, amount: int, address: str):
        logger.debug(f"Transferring {amount}i to address {address}")
        account = self.account_manager.get_account(self.alias)
        account.sync().execute()
        transfer = iw.Transfer(
            amount=amount,
            address=address,
            remainder_value_strategy='ReuseAddress'
        )
        node_response = account.transfer(transfer)
        logger.debug(f"Transferring {amount}i to address {address} ... Ok!")
        return node_response

    def transfer_tokens_multi_address(self, transfer_list):
        logger.debug("Creating multiple transfer ops")
        logger.debug(transfer_list)
        account = self.account_manager.get_account(self.alias)
        account.sync().execute()
        try:
            transfer = iw.TransferWithOutputs(
                outputs=transfer_list,
                remainder_value_strategy="ReuseAddress"
            )
            node_response = account.transfer_with_outputs(transfer)
            logger.debug("Creating multiple transfer ops... Ok!")
            return node_response
        except ValueError as ex:
            message = ex.args[0]
            if "insufficient funds" in message:
                errors = {"message": ex.args[0]}
                raise InsufficientFundsException(message=message,
                                                 errors=errors)

    def restore(self, user):
        pass

    def backup(self, user):
        pass
