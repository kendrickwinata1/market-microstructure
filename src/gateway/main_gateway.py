import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime
import os
import time
from enum import Enum
from threading import Thread
from urllib.parse import urlencode
from pathlib import Path

import requests
import websockets
from binance import AsyncClient, Client
from dotenv import load_dotenv

# Configure logging for the module
logging.basicConfig(
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    level=logging.INFO,
)

# Enum representing the side of an order or trade
class Side(Enum):
    BUY = 0
    SELL = 1

# Enum representing the execution type of an order
class ExecutionType(Enum):
    NEW = 0
    CANCELED = 1
    CALCULATED = 2
    EXPIRED = 3
    TRADE = 4

# Enum representing the status of an order
class OrderStatus(Enum):
    PENDING_NEW = 0  # sent to exchange but has not received any status
    NEW = 1          # order accepted by exchange but not processed yet by the matching engine
    OPEN = 2         # order accepted by exchange and is active on order book
    CANCELED = 3     # order is cancelled
    PARTIALLY_FILLED = 4  # order is partially filled
    FILLED = 5           # order is fully filled and closed (i.e. not expecting any more fills)
    PENDING_CANCEL = 6   # cancellation sent to exchange but has not received any status
    FAILED = 7           # order failed

# Stores and displays order event information
class OrderEvent:
    def __init__(
        self,
        contract_name: str,
        order_id: str,
        execution_type: ExecutionType,
        status: OrderStatus,
        canceled_reason=None,
        client_id=None,
    ):
        self.contract_name = contract_name
        self.order_id = order_id
        self.client_id = client_id
        self.execution_type = execution_type
        self.status = status
        self.canceled_reason = canceled_reason

        # These fields are filled after the event is matched with an execution/trade update
        self.side = None
        self.order_type = None
        self.limit_price = None
        self.last_filled_time = None
        self.last_filled_price = 0
        self.last_filled_quantity = 0

    def __str__(self):
        return (
            f"Order events [contract={self.contract_name}, order_id={self.order_id}, "
            f"status={self.status}, type={self.execution_type}, side={self.side}, "
            f"last_filled_price={self.last_filled_price}, last_filled_qty={self.last_filled_quantity}, "
            f"canceled_reason={self.canceled_reason}]"
        )

    def __repr__(self):
        return str(self)

# Main class for interacting with Binance API for trade execution and listening to updates
class TradeExecutor:
    """
    Acts as the interface between trading logic and Binance exchange.
    Executes trades, logs them, and listens for order execution updates.
    Can also invoke registered callbacks on executions.
    """
    def __init__(self, manager, api_key, api_secret, name="Binance", testnet=True):
        self.manager = manager
        # Filename for logging with timestamp
        self.log_filename = datetime.now().strftime("logs_%Y-%m-%d_%H-%M-%S.txt")

        self.api_key = api_key
        self.api_secret = api_secret
        print("CHECK MY API_KEY: ", api_key)
        print("CHECK MY API_SECRET: ", api_secret)
        self.testnet = testnet
        self._exchange_name = name
        # Callbacks for order execution events
        self._exec_callbacks = []

        # This event loop/thread is used for all async tasks (e.g., websocket listening)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = Thread(target=self._run_async_tasks, daemon=True, name=name)

    def signature(self, data: dict, secret: str) -> str:
        """
        Generate HMAC SHA256 signature for authenticated Binance requests.
        """
        postdata = urlencode(data)
        message = postdata.encode()
        byte_key = bytes(secret, "UTF-8")
        mac = hmac.new(byte_key, message, hashlib.sha256).hexdigest()
        return mac

    def execute_trade(self, trade: dict, direction: str):
        """
        Executes a trade (buy/sell) or cancels an order, if within risk policy.
        Returns True if request was made, else False.
        """
        api_url = "https://testnet.binancefuture.com"
        uri_path = "/fapi/v1/order"
        headers = {"X-MBX-APIKEY": self.api_key}
        signature = self.signature(trade, self.api_secret)

        if direction == "trade":
            # Payload must include all required fields (symbol, side, type, etc.)
            payload = {**trade, "signature": signature}
            req = requests.post(
                api_url + uri_path, headers=headers, data=payload, timeout=1
            )
            result = req.json()
            if "code" not in result.keys():
                self.log_trade_execution(result, "submitted")
            else:
                self.log_trade_execution(result["msg"], "failed")
            print("++++++++++++ATTEMPT TRADE++++++++++++++", result)
            return True

        elif direction == "cancel":
            # For cancelling an order, payload includes orderId and signature
            params = {**trade, "signature": signature}
            req = requests.delete(api_url + uri_path, params=params, headers=headers)
            result = req.json()
            if "code" not in result.keys():
                self.log_trade_execution(result, "submitted")
            else:
                self.log_trade_execution(result["msg"], "failed")
            print("++++++++++++ATTEMPT CANCEL++++++++++++", result)
            return True

    def write_log(self, message, log_directory="./logs"):
        """
        Write log messages to a dated log file for audit or later analysis.
        """
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        filepath = os.path.join(log_directory, self.log_filename)
        with open(filepath, 'a') as file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.write(f"{timestamp} - {message}\n")

    def log_trade_execution(self, result, status):
        """
        Log execution info for submitted/failed/filled orders for auditing.
        """
        if status == "submitted":
            order_id = result.get("orderId")
            symbol = result.get("symbol")
            status = result.get("status")
            side = result.get("side")
            type = result.get("type")

            order_event = OrderEvent(
                symbol, order_id, ExecutionType[status], OrderStatus[status]
            )
            order_event.side = Side[side]
            if type == "MARKET":
                order_event.last_filled_quantity = result.get("origQty")
                logging.info(order_event)
                return True
            elif type == "LIMIT":
                price = result.get("price")
                order_event.limit_price = price
                logging.info(order_event)
                return True

        elif status == "failed":
            print("********FAILED******")
            logging.info("ORDER FAILED : {}".format(result))
            self.write_log(result)
            return True

        elif status == "filled":
            print("********FILLED******")
            logging.info("ORDER FILLED : {}".format(result))
            self.write_log(result)
            return True

    def connect(self):
        """
        Initializes exchange connection and starts event loop for async tasks (e.g., websocket listening).
        """
        logging.info("Initializing connection")
        self._loop.run_until_complete(self._reconnect_ws())
        logging.info("starting event loop thread")
        self._loop_thread.start()
        # Synchronous (REST) client for standard Binance API calls
        self._client = Client(self.api_key, self.api_secret, testnet=self.testnet)

    # Internal: Create a new async client for websocket-based methods
    async def _reconnect_ws(self):
        logging.info("reconnecting websocket")
        self._async_client = await AsyncClient.create(
            self.api_key, self.api_secret, testnet=self.testnet
        )

    # Internal: Runs all async tasks in its own thread/event loop
    def _run_async_tasks(self):
        """Runs the execution listener in this thread."""
        self._loop.create_task(self._listen_execution_forever())
        self._loop.run_forever()

    async def _listen_execution_forever(self):
        """
        Listens for user execution/trade update stream via Binance websocket (user data stream).
        Invokes registered callbacks on each execution event.
        """
        logging.info("Subscribing to user data events")
        _listen_key = await self._async_client.futures_stream_get_listen_key()
        if self.testnet:
            url = "wss://stream.binancefuture.com/ws/" + _listen_key
        else:
            url = "wss://fstream.binance.com/ws/" + _listen_key

        conn = websockets.connect(url)
        ws = await conn.__aenter__()
        while ws.open:
            message = await ws.recv()
            data = json.loads(message)
            update_type = data.get("e")

            if update_type == "ORDER_TRADE_UPDATE":
                trade_data = data.get("o")
                order_id = trade_data.get("i")
                symbol = trade_data.get("s")
                execution_type = trade_data.get("x")
                order_status = trade_data.get("X")
                side = trade_data.get("S")
                last_filled_price = float(trade_data.get("L"))
                last_filled_qty = float(trade_data.get("l"))

                order_event = OrderEvent(
                    symbol,
                    order_id,
                    ExecutionType[execution_type],
                    OrderStatus[order_status],
                )
                order_event.side = Side[side]
                if execution_type == "TRADE":
                    order_event.last_filled_price = last_filled_price
                    order_event.last_filled_quantity = last_filled_qty
                    self.log_trade_execution(order_event, "filled")

                # Trigger all registered callbacks
                if self._exec_callbacks:
                    for _callback in self._exec_callbacks:
                        print(
                            "****************** EXECUTING CALLBACK ******************"
                        )
                        _callback()

    def register_exec_callback(self, callback):
        """
        Register a callback function that will be called whenever a trade/order execution event is received.
        """
        print("################ REGISTERING CALLBACK ################")
        self._exec_callbacks.append(callback)

# ----------- Load Environment Variables -----------
PROJECT_DIR = Path(__file__).parent.parent.resolve()
ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
logging.info(f"API_KEY: {API_KEY}")
logging.info(f"API_SECRET: {API_SECRET}")

# ----------- Script Demo -----------
if __name__ == "__main__":
    api_key = API_KEY
    api_secret = API_SECRET

    # Connect to Binance and start event loop
    TradeExecutor("", api_key, api_secret).connect()
    time.sleep(5)

    # Example trade order data for demo purposes
    data = {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "type": "MARKET",
        "quantity": 0.002,
        "timestamp": int(time.time() * 1000),
        "recvWindow": 60000,
    }
    cancel_data = {
        "orderId": 4036972677,
        "symbol": "BTCUSDT",
        "timestamp": int(time.time() * 1000),
        "recvWindow": 60000,
    }
    trader = TradeExecutor("", api_key, api_secret)
    while True:
        time.sleep(1)
        trader.execute_trade(data, "trade")
