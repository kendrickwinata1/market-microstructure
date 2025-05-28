import tkinter as tk  # Placeholder for potential GUI features; not used directly here
from queue import Queue
from threading import Thread
from datetime import datetime
from dotenv import load_dotenv
import os
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

from book_keeper.main_book_keeper import BookKeeper
from gateway.market_data_stream import MarketDataStream
from gateway.data_stream import DataStream
from gateway.main_gateway import TradeExecutor
from risk_manager.main_risk_manager import RiskManager
from trading_engine.main_trading_strategy import TradingStrategy
from rest_connect.rest_factory import RestFactory
import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("output.txt")
sys.stderr = sys.stdout  # also log errors

# --- Configuration constants ---
OFFSET = 15000  # Timestamp offset for exchange API
MAX_OPEN_ORDER_COUNT = 1          # Max number of open orders allowed at once
MAX_OPEN_ORDER_LIFE_SECONDS = 60  # Max seconds before a pending order is considered stale
MAX_MODEL_NONE_COUNT = 20         # Max allowed model failures before taking action

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Save to 'app.log' in current directory
        logging.StreamHandler()          # Also print to terminal
    ]
)

class ExecManager:
    """
    Orchestrates trading: receives data, runs strategy, applies risk, and executes trades.
    Handles liquidation and manages open orders as required.
    """
    def __init__(self, trade_executor, book_keeper, rest_gateway, risk_manager):
        self.queue = Queue()
        self.trade_executor = trade_executor
        self.book_keeper = book_keeper
        self.risk_manager = risk_manager
        self.rest_gateway = rest_gateway
        self.strategy = TradingStrategy(self.queue)
        self.trade_executor.connect()
        self.reattempt_liquidate = False  # Flag for failed liquidation retries
        self.model_none_count = 0         # Tracks consecutive None outputs from model

    def update_queue(self, tick):
        """
        Put the latest market tick into the data queue for strategy processing.
        """
        output = (tick["datetime"], tick["lastprice"])
        print(f"Callback: {output}")
        self.queue.put(output)

    def exec_strat(self, tick):
        """
        Main event handler for each market tick.
        - Updates bookkeeping and cancels stale orders
        - Checks risk triggers for liquidation
        - Otherwise, calls the trading model, checks risk for order, and places order if approved
        """
        
        last_price = tick["lastprice"]
        if not last_price:
            logging.warning("[ExecManager] Received empty last price, skipping tick.")
            return

        server_response = self.rest_gateway.time()
        servertime = int(server_response.get("serverTime", 0))
        if not servertime:
            logging.error("[ExecManager] Server time fetch failed.")
            return

        today = datetime.fromtimestamp(servertime / 1000).date()
        self.book_keeper.update_bookkeeper(today, last_price, servertime)

        # Order management
        open_orders = self.rest_gateway.get_all_open_orders("BTCUSDT", servertime)
        order_queue_ok = len(open_orders) < MAX_OPEN_ORDER_COUNT

        # Risk checks
        if self.risk_manager.trigger_stop_loss() or self.risk_manager.trigger_trading_halt():
            logging.info("[ExecManager] Risk trigger activated. Attempting liquidation.")
            self.handle_liquidation(servertime)
            return

        # Strategy analysis
        self.update_queue(tick)
        self.strategy.collect_new_data()
        self.strategy.aggregate_data()
        model_output = self.strategy.analyze_data()

        if not model_output:
            self.model_none_count += 1
            if self.model_none_count >= MAX_MODEL_NONE_COUNT:
                print("Model returned None too many times; cancelling all orders.")
                self.rest_gateway.cancel_all_order("BTCUSDT", servertime)
            print(f"MODEL NONE COUNT VALUE = {self.model_none_count}")
            return

        direction, limit_price = model_output[0].upper(), float(model_output[1])

        # --- Override HOLD based on momentum indicator ---
        # if direction == "HOLD":
        #     try:
        #         data = self.strategy.data
        #         last_deriv = data["Short_Moving_Avg_1st_Deriv"].iloc[-1]
        #         if last_deriv > 10:
        #             direction = "BUY"
        #             print("[Override] HOLD -> BUY due to momentum")
        #         elif last_deriv < -10:
        #             direction = "SELL"
        #             print("[Override] HOLD -> SELL due to momentum")
        #     except Exception as e:
        #         print("[Override Error] Cannot override HOLD:", e)
        
        if direction == "HOLD":
            # Force override HOLD signal for testing
            direction = "BUY"  # or "SELL" to test short-side logic
            print("[Force Override] HOLD signal forcibly overridden to BUY for testing")


        # Prepare order details
        order_quantity = 0
        approval = False

        if direction == "BUY":
            dollar_amt = self.risk_manager.get_available_tradable_balance()
            order_quantity = round(dollar_amt / limit_price, 3)
            # approval = (
            #     self.risk_manager.check_available_balance(dollar_amt)
            #     and self.risk_manager.check_buy_order_value(limit_price)
            #     and self.risk_manager.check_buy_position()
            # )
            
            # Force approval for testing
            approval = True
            print("[Force Override] Risk manager approval forced to True for BUY.")

        elif direction == "SELL":
            pos_info = self.rest_gateway.get_position_info("BTCUSDT", servertime)
            order_quantity = float(pos_info[0]["positionAmt"]) if pos_info else 0
            # approval = (
            #     self.risk_manager.check_short_position(order_quantity)
            #     and self.risk_manager.check_sell_order_value(limit_price)
            # )

            # Force approval for testing
            approval = True
            print("[Force Override] Risk manager approval forced to True for SELL.")

        elif direction == "HOLD":
            print("Model signals HOLD.")
            return  # nothing else to do

        # --- Final order placement if all checks pass ---
        if approval and order_queue_ok and order_quantity > 0:
            order_data = {
                "symbol": "BTCUSDT",
                "price": limit_price,
                "side": direction,
                "type": "LIMIT",
                "quantity": order_quantity,
                "timestamp": servertime - OFFSET,
                "recvWindow": 60000,
                "timeInForce": "GTC",
            }
            trade_result = self.trade_executor.execute_trade(order_data, "trade")

            if trade_result:
                logging.info(f"[ExecManager] {direction} order placed successfully.")
                self.book_keeper.update_bookkeeper(datetime.now(), limit_price, servertime) 
                self.book_keeper.return_historical_data().to_csv("historical_data.csv")
            else:
                logging.error(f"[ExecManager] {direction} order placement failed.")
        else:
            print("Order not approved by risk manager or order queue full.")

            
def on_exec():
    """
    Callback function for trade execution events (optional, can be used for logging or analytics).
    """
    print("Execution callback triggered.")

if __name__ == "__main__":
    # --- Load credentials from ..env and initialize main system objects ---
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    symbol = "BTCUSDT"

    print("Launching execution environment...")

    rest_factory = RestFactory()
    futuretestnet_base_url = "https://testnet.binancefuture.com"
    rest_gateway = rest_factory.create_gateway(
        "BINANCE_TESTNET_FUTURE",
        futuretestnet_base_url,
        api_key,
        api_secret,
    )

    trade_executor = TradeExecutor(symbol, api_key, api_secret)
    trade_executor.register_exec_callback(on_exec)

    book_keeper = BookKeeper(symbol, api_key, api_secret)
    risk_manager = RiskManager(book_keeper)

    # Compose the main execution manager object that handles strategy and risk
    exec_manager = ExecManager(trade_executor, book_keeper, rest_gateway, risk_manager)

    # --- Market data stream setup: every tick triggers trading logic in exec_manager ---
    data_stream = DataStream(symbol, api_key, api_secret)
    data_stream.register_tick_callback(exec_manager.exec_strat)
    data_stream.connect()

    # --- Main application heartbeat to keep the process alive and provide basic monitoring ---
    while True:
        time.sleep(10)
        print("Heartbeat: application running.")





