import datetime
import json
import websocket

class MarketDataStream:
    """
    Handles real-time BTC/USDT trade data streaming from Binance via WebSocket.
    Designed to provide fresh price data to a queue for use in trading strategies,
    helping avoid overloading REST API rate limits.
    """

    def __init__(self, queue):
        """
        Initialize with a thread-safe queue for outputting streaming data.
        :param queue: A thread-safe queue to put (timestamp, price) tuples into.
        """
        self.queue = queue

    def fetch_data(self):
        """
        Starts a WebSocket client in the background to stream live BTC/USDT trades.
        Each new price tick is parsed and pushed as (timestamp, price) into the queue.
        """

        def on_message(ws, message):
            """
            Handles incoming messages from the Binance websocket.
            If a price ('p') is present in the message, puts (timestamp, price) to the queue.
            """
            data = json.loads(message)
            if "p" in data:
                price = float(data["p"])
                timestamp = datetime.datetime.now()
                self.queue.put((timestamp, price))

        def on_open(ws):
            """
            Called once the websocket connection is open.
            Subscribes to the BTCUSDT trade stream.
            """
            params = {
                "method": "SUBSCRIBE",
                "params": ["btcusdt@trade"],
                "id": 1
            }
            ws.send(json.dumps(params))

        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(
            "wss://stream.binance.com:9443/ws/btcusdt@trade",
            on_message=on_message,
            on_open=on_open,
        )
        ws.run_forever()
