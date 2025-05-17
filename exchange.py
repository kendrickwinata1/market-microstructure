import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

class BinanceFuturesExchange:
    def __init__(self, config):
        self.symbol = config.symbol.upper()
        self.api_key = config.api_key
        self.api_secret = config.api_secret.encode()
        self.headers = {"X-MBX-APIKEY": self.api_key}
        self.base_url = "https://testnet.binancefuture.com"
        self.last_order_id = None

    def _sign_params(self, params):
        query_string = urlencode(params)
        signature = hmac.new(self.api_secret, query_string.encode(), hashlib.sha256).hexdigest()
        return {**params, "signature": signature}

    def get_order_book(self, limit=5):
        """Get best bid/ask."""
        url = f"{self.base_url}/fapi/v1/depth"
        response = requests.get(url, params={"symbol": self.symbol, "limit": limit})
        data = response.json()
        best_bid = float(data['bids'][0][0])
        best_ask = float(data['asks'][0][0])
        return best_bid, best_ask

    def place_order(self, side, quantity, price, reduce_only=False):
        url = f"{self.base_url}/fapi/v1/order"
        params = {
            "symbol": self.symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": price,
            "recvWindow": 5000,
            "timestamp": int(time.time() * 1000),
            "reduceOnly": str(reduce_only).lower()
        }
        params = self._sign_params(params)
        response = requests.post(url, headers=self.headers, params=params)
        data = response.json()
        self.last_order_id = data.get("orderId")
        return data

    def cancel_all_orders(self):
        url = f"{self.base_url}/fapi/v1/allOpenOrders"
        params = {
            "symbol": self.symbol,
            "timestamp": int(time.time() * 1000)
        }
        params = self._sign_params(params)
        response = requests.delete(url, headers=self.headers, params=params)
        return response.json()

    def get_balance(self, asset="USDT"):
        url = f"{self.base_url}/fapi/v2/balance"
        params = {"timestamp": int(time.time() * 1000)}
        params = self._sign_params(params)
        response = requests.get(url, headers=self.headers, params=params)
        balances = response.json()
        for entry in balances:
            if entry["asset"] == asset:
                return float(entry["availableBalance"])
        return 0.0

    # Add any extra utility for backtest mode as needed
