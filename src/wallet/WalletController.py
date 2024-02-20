import os
from payment.PaymentGateway.IOTAPayment.IOTAPaymentController import IOTAPaymentController
from payment.PaymentGateway.IOTAPayment.IOTAPaymentController import WalletConfig
from conf import settings


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


class WalletController:
    local_pow = True
    alias = settings.WALLET_NAME

    def __init__(self):
        self.controller = IOTAPaymentController(config=wallet_config())

    def create_wallet(self, store_mnemonic=False):
        pass

    def create_account(self):
        return self.controller.create_account(identifier=self.alias)

    def get_balance(self):
        return self.controller.get_balance(identifier=self.alias)

    def get_address(self):
        return self.controller.get_address(email=self.alias)

    def get_transaction_list(self):
        return self.controller.get_transaction_history(identifier=self.alias)

    def transfer_tokens(self, amount: int, address: str):
        return self.controller.execute_transaction(from_identifier=self.alias, to_identifier=address, value=amount)

    def transfer_tokens_multi_address(self, transfer_list):
        # https://github.com/iotaledger/iota-sdk/blob/develop/bindings/python/examples/wallet/12-prepare_output.py
        account = self.controller.wallet.get_account(self.alias)
        account.sync()
        params = []
        for transfer in transfer_list:
            params.append({
                "address": transfer['address'],
                "amount": str(transfer['amount']),
            })

        transaction = account.send_with_params(params)
        return transaction

    def restore(self, user):
        pass

    def backup(self, user):
        pass
