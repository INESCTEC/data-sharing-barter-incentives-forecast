import os

from pprint import pprint
from loguru import logger
from dotenv import load_dotenv

load_dotenv('.env')

from conf import settings
from core.wallet import WalletController
from core import MarketController

from core.api.exception.APIException import (
    NoMarketSessionException,
    MarketSessionException,
)

# logger:
format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}"
logger.add("files/logfile.log", format=format, level='DEBUG', backtrace=True)
logger.info("-" * 79)


def main_no_installation():
    while True:
        _clear_console()
        print("     MAIN MENU - No Wallet Detected")
        print("1  - New Installation")
        _sep()
        print("0 - Exit")
        _empty()
        choice = input("Please make a choice: ")

        if choice == "1":
            installation_menu()
            return
        elif choice == "0":
            exit("Exit.")
        else:
            print("Invalid option.")


def main():
    while True:
        _clear_console()
        print("     MARKET MAIN MENU - Wallet Detected")
        print("1  - Market Operations")
        print("2  - Wallet Operations")
        _sep()
        print("0 - Exit")
        _empty()
        choice = input("Please make a choice: ")

        if choice == "1":
            market_menu()
        if choice == "2":
            wallet_menu()
        elif choice == "0":
            exit("Exit.")
        else:
            print("Invalid option.")


def installation_menu():
    _clear_console()
    print("This will create a new market wallet & account.")
    choice = input("Proceed? (Y/n)")
    if choice.lower() == "y":
        wallet = WalletController()
        wallet.create_wallet(store_mnemonic=True)
        wallet.create_account()
        address = wallet.get_address()
        print("Market Wallet address (use it to transfer tokens):")
        print(address)

    input("Press any key to continue.")
    return


def market_menu():
    market = MarketController()

    while True:
        _clear_console()
        print("     Market OPS MENU")
        print("1  - Open market session")
        print("2  - Get bids for current market session")
        print("3  - Approve market bids")
        print("4  - Close market session")
        print("5  - Run market session")
        print("6  - Transfer token balance back to agents")
        _sep()
        print("9 - Return to previous menu.")
        print("0 - Exit")
        _empty()
        choice = input("Please make a choice: ")

        if choice == "1":
            try:
                # Create first market session:
                market.open_market_session()
            except NoMarketSessionException:
                logger.error("Failed! There are no 'staged' market sessions")
        if choice == "2":
            try:
                # Create first market session:
                bids = market.get_buyers_bids()
                pprint(bids)
            except Exception as ex:
                logger.exception("Failed! Unable to retrieve market bids")
        elif choice == "3":
            # Approve buyers bids:
            try:
                market.approve_buyers_bids()
            except Exception as ex:
                logger.exception(repr(ex))
        elif choice == "4":
            # Close market session (no more bids):
            try:
                market.close_market_session()
            except Exception as ex:
                logger.exception(repr(ex))
        elif choice == "5":
            # Run market session:
            try:
                market.run_market_session()
            except Exception as ex:
                logger.exception(repr(ex))
        elif choice == "6":
            try:
                # Transfer tokens back to clients:
                market.transfer_tokens_out()
            except MarketSessionException as ex:
                logger.exception(repr(ex))
        elif choice == "9":
            return
        elif choice == "0":
            exit("Exit.")
        else:
            print("Invalid option.")

        input("Press any key to continue.")


def wallet_menu():
    wallet = WalletController()

    while True:
        _clear_console()
        print("     Wallet OPS MENU")
        print("1  - Get wallet address")
        print("2  - Get wallet balance")
        print("3  - Transfer balance to address")
        _sep()
        print("9 - Return to previous menu.")
        print("0 - Exit")
        _empty()
        choice = input("Please make a choice: ")

        if choice == "1":
            try:
                address = wallet.get_address()
                print(f"Wallet Address: {address}")
            except Exception as ex:
                logger.exception(repr(ex))
        elif choice == "2":
            # Approve buyers bids:
            try:
                balance = wallet.get_balance()
                print(f"Wallet Balance: {balance}i")
            except Exception as ex:
                logger.exception(repr(ex))
        elif choice == "3":
            # Close market session (no more bids):
            try:
                amount = input("Enter transfer amount "
                               "(use 'FB' keywork for full balance "
                               "transfer): ")
                if amount.lower() == "fb":
                    amount = wallet.get_balance()["available"]
                else:
                    amount = int(amount)
                out_address = input("Enter output address: ")
                # -- initialize WALLET controller:
                node_response = wallet.transfer_tokens(
                    amount=amount,
                    address=out_address
                )
                print("Node Response:", node_response)
            except Exception as ex:
                logger.exception(repr(ex))
        elif choice == "9":
            return
        elif choice == "0":
            exit("Exit.")
        else:
            print("Invalid option.")

        input("Press any key to continue.")


def _clear_console():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def _empty():
    print("")


def _sep():
    print("===========================================")


if __name__ == '__main__':
    wallet_path = os.path.join(settings.WALLET_STORAGE_PATH, "wallet-db")
    if not os.path.exists(wallet_path):
        main_no_installation()
    else:
        main()
