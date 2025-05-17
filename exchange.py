# exchange.py
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException


class BinanceTestnetExchange:
    def __init__(self, config):
        self.client = Client(config.api_key, config.api_secret, testnet=True)
        self.symbol = config.symbol
        self.last_bid_price = None
        self.last_ask_price = None
        logging.info("Connected to Binance Testnet for symbol %s", self.symbol)

    def get_fair_price(self):
        try:
            ob = self.client.get_order_book(symbol=self.symbol, limit=5)
            best_bid = float(ob['bids'][0][0])
            best_ask = float(ob['asks'][0][0])
            self.last_bid_price = best_bid
            self.last_ask_price = best_ask
            return (best_bid + best_ask) / 2.0
        except BinanceAPIException as e:
            logging.error("Failed to fetch order book: %s", e)
            return None

    def place_order(self, side, price, quantity):
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type="LIMIT",
                timeInForce="GTC",
                quantity=quantity,
                price=f"{price:.2f}"
            )
            logging.info("Placed %s order: %.4f BTC @ %.2f", side, quantity, price)
            return order
        except BinanceAPIException as e:
            logging.error("Order placement failed: %s", e)
            return None

    def cancel_all_orders(self):
        try:
            orders = self.client.get_open_orders(symbol=self.symbol)
            for order in orders:
                self.client.cancel_order(symbol=self.symbol, orderId=order['orderId'])
            if orders:
                logging.info("Canceled %d open orders", len(orders))
        except BinanceAPIException as e:
            logging.error("Failed to cancel orders: %s", e)

    def get_inventory(self, asset):
        """Return balance for a specific asset, e.g., 'BTC' or 'USDT'."""
        account = self.client.get_account()
        for bal in account['balances']:
            if bal['asset'] == asset:
                return float(bal['free'])
        return 0.0


# For backtesting, see strategy.py (SimulatedExchange class is defined there for simplicity)
