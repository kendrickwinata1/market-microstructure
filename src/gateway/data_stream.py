import asyncio
import logging
from threading import Thread
import datetime
import os
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager, Client

# Set up logging with thread and log level information
logging.basicConfig(
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    level=logging.INFO,
)

class DataStream:
    """
    Handles real-time streaming of market data from Binance using websockets.
    Uses an event loop in a separate thread to fetch both trade and book ticker streams.
    Notifies registered callback functions with the latest tick data.
    """

    def __init__(self, symbol: str, api_key=None, api_secret=None):
        self._api_key = api_key
        self._api_secret = api_secret
        self._symbol = symbol

        # Binance client instances (sync and async)
        self._client = None
        self._async_client = None
        self._binance_socket_manager = None
        self._multi_socket = None  # Binance multi-stream websocket

        # Most recent market data cache
        self._market_cache = None

        # Async event loop & thread for streaming
        self._loop = asyncio.new_event_loop()
        self._loop_thread = Thread(target=self._run_async_tasks, daemon=True)

        # List of user-registered callback functions for tick updates
        self._tick_callbacks = []

        # Output dictionary to hold current market tick information
        self.output = {
            "lastprice": "",
            "lastquantity": "",
            "bestbidprice": "",
            "bestbidquantity": "",
            "bestaskprice": "",
            "bestaskquantity": "",
        }

    def connect(self):
        """
        Initializes async websocket connection and starts the background event loop.
        Also creates the synchronous Binance client for non-async REST calls.
        """
        logging.info("Initializing connection")
        self._loop.run_until_complete(self._reconnect_ws())
        logging.info("starting event loop thread")
        self._loop_thread.start()
        self._client = Client(self._api_key, self._api_secret)

    async def _reconnect_ws(self):
        """
        Internal async method to create a new async Binance client.
        Forces use of testnet for safety.
        """
        logging.info("reconnecting websocket")
        self._async_client = await AsyncClient.create(
            self._api_key, self._api_secret, testnet=True
        )

    def _run_async_tasks(self):
        """
        Runs the async market data listening task in the event loop of this thread.
        """
        self._loop.create_task(self._listen_market_forever())
        self._loop.run_forever()

    async def _listen_market_forever(self):
        """
        Subscribes to Binance trade and bookTicker websockets, parses new data,
        and calls registered callbacks with every new tick.
        """
        logging.info("Subscribing to depth events")

        while True:
            if not self._multi_socket:
                logging.info("depth socket not connected, reconnecting")
                self._binance_socket_manager = BinanceSocketManager(self._async_client)
                # Subscribe to both trade and bookTicker multiplexed futures streams
                self._multi_socket = (
                    self._binance_socket_manager.futures_multiplex_socket(
                        [
                            self._symbol.lower() + "@trade",
                            self._symbol.lower() + "@bookTicker"
                        ]
                    )
                )

            try:
                async with self._multi_socket as ms:
                    # Wait for new message from Binance websocket
                    self._market_cache = await ms.recv()

                    # Update output dictionary depending on stream type
                    if "@trade" in self._market_cache["stream"]:
                        self.output["lastprice"] = self._market_cache["data"]["p"]
                        self.output["lastquantity"] = self._market_cache["data"]["q"]
                    else:
                        self.output["bestbidprice"] = self._market_cache["data"]["b"]
                        self.output["bestbidquantity"] = self._market_cache["data"]["B"]
                        self.output["bestaskprice"] = self._market_cache["data"]["a"]
                        self.output["bestaskquantity"] = self._market_cache["data"]["A"]

                    self.output["datetime"] = datetime.datetime.now()

                    # Notify all registered callbacks with new tick
                    if self._tick_callbacks:
                        for _callback in self._tick_callbacks:
                            _callback(self.output)

            except Exception as e:
                logging.exception("encountered issue in depth processing")
                # On error, reset socket and force reconnect
                self._multi_socket = None
                await self._reconnect_ws()

    def register_tick_callback(self, callback):
        """
        Register a callback function to be called with every new market tick.
        """
        self._tick_callbacks.append(callback)

# Example callback for debug/testing
def on_tick(s):
    return s

if __name__ == "__main__":
    # --- Load API keys from .env file ---
    # Looks for .env in current or parent directories
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("API_KEY or API_SECRET not set in .env!")

    # Create and start the streamer
    streamer = DataStream("BTCUSDT", api_key, api_secret)
    streamer.register_tick_callback(on_tick)
    streamer.connect()

    while True:
        pass
