import os

from pprint import pprint
from loguru import logger
from dotenv import load_dotenv

load_dotenv('.env')

from conf import settings
from src.wallet import WalletController
from src.MarketController import MarketController

from src.api.exception.APIException import (
    NoMarketSessionException,
    MarketSessionException,
    MarketWalletAddressException,
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
        print("2  - Market Configuration")
        print("3  - Wallet Operations")
        _sep()
        print("0 - Exit")
        _empty()
        choice = input("Please make a choice: ")

        if choice == "1":
            market_menu()
        if choice == "2":
            market_configuration()
        if choice == "3":
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
        address = wallet.get_address()["address"]["inner"]
        print("Market Wallet address (use it to transfer tokens):")
        print(address)

    input("Press any key to pass.")
    return


def market_configuration():
    try:
        market = MarketController()
    except Exception:
        logger.exception("Unable to login to the platform")
        input("Press any key to return to main menu.")
        return

    while True:
        _clear_console()
        print("     Market Config MENU")
        print("1  - Register market wallet address")
        print("2  - Get current market wallet address")
        print("3  - Update market wallet address")
        _sep()
        print("9 - Return to previous menu.")
        print("0 - Exit")
        _empty()
        choice = input("Please make a choice: ")

        if choice == "1":
            try:
                address = input("Enter market wallet address: ")
                market.register_market_wallet_address(address=address)
            except MarketWalletAddressException:
                pass
        if choice == "2":
            try:
                market.get_market_wallet_address()
            except MarketWalletAddressException:
                logger.error("Failed to get wallet address.")
        if choice == "3":
            try:
                old_address = input("Enter market wallet address (old): ")
                new_address = input("Enter market wallet address (new): ")
                market.update_market_wallet_address(
                    old_address=old_address,
                    new_address=new_address
                )
            except MarketWalletAddressException:
                pass
        elif choice == "9":
            return
        elif choice == "0":
            exit("Exit.")
        else:
            print("Invalid option.")

        input("Press any key to pass.")


def market_menu():
    try:
        market = MarketController()
    except Exception:
        logger.exception("Unable to login to the platform")
        input("Press any key to return to main menu.")
        return

    while True:
        _clear_console()
        print("     Market OPS MENU")
        print("1  - Open market session")
        print("2  - Get bids for current market session")
        print("3  - Approve market bids")
        print("4  - Close market session")
        print("5  - Run market session")
        print("6  - Get users market balance")
        print("7  - Transfer token balance back to agents")
        print("8  - Validate token transfers")
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
                pass
            except MarketSessionException:
                pass
        if choice == "2":
            try:
                # Create first market session:
                bids = market.get_buyers_bids()
                pprint(bids)
            except Exception:
                pass
        elif choice == "3":
            # Approve buyers bids:
            try:
                market.approve_buyers_bids()
            except Exception:
                pass
        elif choice == "4":
            # Close market session (no more bids):
            try:
                market.close_market_session()
            except Exception:
                pass
        elif choice == "5":
            # Run market session:
            try:
                market.run_market_session()
            except Exception:
                pass
        elif choice == "6":
            try:
                # List users market balance:
                market.list_user_market_balance()
            except Exception:
                pass
        elif choice == "7":
            try:
                # Transfer tokens back to clients:
                market.transfer_tokens_out()
            except MarketSessionException:
                pass
        elif choice == "8":
            try:
                # Transfer tokens back to clients:
                market.validate_tokens_transfer()
            except MarketSessionException:
                pass
        elif choice == "9":
            return
        elif choice == "0":
            exit("Exit.")
        else:
            print("Invalid option.")

        input("Press any key to pass.")


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
                address = wallet.get_address()["address"]["inner"]
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

        input("Press any key to pass.")


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
    main()
