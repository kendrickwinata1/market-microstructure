import tkinter as tk
from queue import Queue
from threading import Thread
from datetime import datetime
from dotenv import load_dotenv
import os
import time
import logging
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading


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
from gateway.data_stream import DataStream
from gateway.main_gateway import TradeExecutor
from risk_manager.main_risk_manager import RiskManager
from trading_engine.main_trading_strategy import TradingStrategy
from rest_connect.rest_factory import RestFactory
from visualization.live_plotter import live_performance_plot
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

# ANSI color codes for terminal output (works in most terminals)
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

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
MAX_OPEN_ORDER_COUNT = 1  # Max number of open orders allowed at once
MAX_OPEN_ORDER_LIFE_SECONDS = 60  # Max seconds before a pending order is considered stale
MAX_MODEL_NONE_COUNT = 20  # Max allowed model failures before taking action


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
        self.model_none_count = 0  # Tracks consecutive None outputs from model

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
        - Updates bookkeeping and cancels stale orders.
        - Calls the trading model and conditionally overrides HOLD based on strong signals.
        - Checks risk and executes trades.
        """
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        logging.info("[ExecManager] Received new tick data.")

        last_price = tick.get("lastprice")
        print(f"\n{CYAN}{'='*18} NEW TICK {'='*18}{RESET}")
        print(f"{YELLOW}Tick received at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Last price: {last_price}{RESET}")

        if last_price == "":
            print(f"{RED}No last price received; skipping tick.{RESET}")
            logging.warning("[ExecManager] Empty last price received; skipping tick.")
            return

        # 1. Get server time and date
        server_response = self.rest_gateway.time()
        servertime = int(server_response.get("serverTime", 0))
        if not servertime:
            print(f"{RED}Failed to fetch server time; aborting tick processing.{RESET}")
            logging.error("[ExecManager] Failed to fetch server time; aborting tick processing.")
            return

        today = datetime.fromtimestamp(servertime / 1000).date()
        print(f"{CYAN}Updating bookkeeper for date: {today}, price: {last_price}...{RESET}")
        logging.info(f"[ExecManager] Updating bookkeeper for date: {today}, price: {last_price}")
        self.book_keeper.update_bookkeeper(today, last_price, servertime)

        # Order management
        open_orders = self.rest_gateway.get_all_open_orders("BTCUSDT", servertime)
        order_queue_ok = len(open_orders) < MAX_OPEN_ORDER_COUNT
        print(f"{CYAN}Open orders:{RESET} {len(open_orders)} | Queue OK: {order_queue_ok}")
        logging.info(f"[Order Management] Open orders: {len(open_orders)}; Queue OK: {order_queue_ok}")

        # Risk trigger check
        stop_loss = self.risk_manager.trigger_stop_loss()
        trading_halt = self.risk_manager.trigger_trading_halt()
        print(f"{CYAN}Risk check:{RESET} stop_loss: {stop_loss}, trading_halt: {trading_halt}")
        logging.info(f"[Risk Manager] stop_loss: {stop_loss}, trading_halt: {trading_halt}")
        if stop_loss or trading_halt:
            print(f"{RED}Stop-loss or trading halt triggered; initiating liquidation.{RESET}")
            logging.info("[Risk Manager] Stop-loss or trading halt triggered; initiating liquidation.")
            # self.handle_liquidation(servertime) # If implemented
            return

        # Model analysis
        print(f"{CYAN}Collecting new data and aggregating features...{RESET}")
        logging.info("[Strategy] Collecting new data and aggregating features.")
        self.update_queue(tick)
        self.strategy.collect_new_data()
        self.strategy.aggregate_data(save_csv=True)

        model_output = self.strategy.analyze_data()
        print(f"{CYAN}Model output:{RESET} {model_output}")
        logging.info(f"[Strategy] Model output: {model_output}")
        
        # <<<<<<<< PRINT REPORT >>>>>>>>
        self.print_performance_report()

        # Check if model output is valid
        if not model_output:
            self.model_none_count += 1
            print(f"{YELLOW}Model returned None ({self.model_none_count}/{MAX_MODEL_NONE_COUNT}){RESET}")
            logging.warning(f"[Strategy] Model returned None ({self.model_none_count}/{MAX_MODEL_NONE_COUNT}).")
            if self.model_none_count >= MAX_MODEL_NONE_COUNT:
                print(f"{RED}Model failure limit reached; cancelling all orders.{RESET}")
                logging.warning("[ExecManager] Model failure limit reached; cancelling all orders.")
                self.rest_gateway.cancel_all_order("BTCUSDT", servertime)
            return
        else:
            self.model_none_count = 0  # Reset on valid signal

        # Get direction and limit_price from the model/strategy
        direction, limit_price = model_output[0].upper(), float(model_output[1])
        print(f"{GREEN}Signal: {direction} | Limit price: {limit_price}{RESET}")
        logging.info(f"[Strategy] Initial model signal: {direction} at price: {limit_price}")

        # Handle HOLD signals with adaptive overrides based on strategy metrics
        if direction == "HOLD":
            data = self.strategy.data
            if "Short_Moving_Avg_1st_Deriv" in data.columns and "KalmanFilterEst_1st_Deriv" in data.columns:
                short_ma_deriv = data["Short_Moving_Avg_1st_Deriv"].iloc[-1]
                kalman_deriv = data["KalmanFilterEst_1st_Deriv"].iloc[-1]

                print(f"{YELLOW}Short_MA_1st_Deriv: {short_ma_deriv} | Kalman_1st_Deriv: {kalman_deriv}{RESET}")
                logging.debug(f"[Strategy Debug] Short MA Derivative: {short_ma_deriv}, Kalman Derivative: {kalman_deriv}")

                if short_ma_deriv > 10 and kalman_deriv > 0:
                    direction = "BUY"
                    print(f"{YELLOW}[Override] HOLD overridden to BUY due to positive momentum.{RESET}")
                    logging.info("[Strategy Override] HOLD overridden to BUY due to positive momentum.")
                elif short_ma_deriv < -10 and kalman_deriv < 0:
                    direction = "SELL"
                    print(f"{YELLOW}[Override] HOLD overridden to SELL due to negative momentum.{RESET}")
                    logging.info("[Strategy Override] HOLD overridden to SELL due to negative momentum.")
                else:
                    print(f"{YELLOW}HOLD confirmed; insufficient momentum for override.{RESET}")
                    logging.info("[Strategy] HOLD signal confirmed; insufficient momentum for override.")
                    return  # Exit without placing an order
            else:
                print(f"{YELLOW}HOLD override check skipped: Required feature columns not available.{RESET}")
                logging.warning("[Strategy] HOLD override check skipped: Required feature columns not available.")
                return
    

        # Determine order quantity and perform risk checks
        order_quantity = 0
        approval = False
        action = "NONE"

        # Get current position size once to determine action (open/close)
        pos_info = self.rest_gateway.get_position_info("BTCUSDT", servertime)
        current_btc_pos = 0.0
        for pos in pos_info:
            if pos.get('symbol') == 'BTCUSDT':
                current_btc_pos = float(pos.get('positionAmt', 0))
                break
        print(f"{CYAN}Current BTC position: {current_btc_pos}{RESET}")
        logging.info(f"[Position Check] Current position size: {current_btc_pos}")

        if direction == "BUY":
            if current_btc_pos < 0:
                action = "CLOSE_SHORT"
                order_quantity = abs(current_btc_pos)
                approval = (
                    self.risk_manager.check_buy_order_value(limit_price) and
                    self.risk_manager.check_buy_to_cover_value(limit_price)
                )
                print(f"{GREEN}[Risk Check] {action} approval: {approval}, Quantity: {order_quantity}{RESET}")
                logging.info(f"[Risk Check] {action} approval: {approval}, Quantity: {order_quantity}")

            elif current_btc_pos == 0:
                action = "OPEN_LONG"
                dollar_amt = self.risk_manager.get_available_tradable_balance()
                print(f"{CYAN}Available to open long: {dollar_amt}{RESET}")
                if dollar_amt < 10:
                    print(f"{YELLOW}[Risk Check] Available dollar amount {dollar_amt:.2f} too small for a BUY order; skipping.{RESET}")
                    logging.warning(f"[Risk Check] Available dollar amount {dollar_amt:.2f} too small for a BUY order; skipping.")
                else:
                    order_quantity = round(dollar_amt / limit_price, 3)
                    approval = (
                        self.risk_manager.check_available_balance(dollar_amt)
                        and self.risk_manager.check_buy_order_value(limit_price)
                        and self.risk_manager.check_buy_position()
                    )
                print(f"{GREEN}[Risk Check] {action} approval: {approval}, Amount: {dollar_amt:.2f}, Quantity: {order_quantity}{RESET}")
                logging.info(f"[Risk Check] {action} approval: {approval}, Amount: {dollar_amt:.2f}, Quantity: {order_quantity}")

        elif direction == "SELL":
            if current_btc_pos > 0:
                action = "CLOSE_LONG"
                order_quantity = current_btc_pos
                approval = (
                    order_quantity > 0
                    and self.risk_manager.check_short_position(order_quantity)
                    and self.risk_manager.check_sell_order_value(limit_price)
                )
                print(f"{RED}[Risk Check] {action} approval: {approval}, Position quantity: {order_quantity}{RESET}")
                logging.info(f"[Risk Check] {action} approval: {approval}, Position quantity: {order_quantity}")

            elif current_btc_pos == 0:
                action = "OPEN_SHORT"
                dollar_amt = self.risk_manager.get_available_tradable_balance()
                print(f"{CYAN}Available to open short: {dollar_amt}{RESET}")
                if dollar_amt < 10:
                    print(f"{YELLOW}[Risk Check] Available dollar amount {dollar_amt:.2f} too small for a SELL order; skipping.{RESET}")
                    logging.warning(f"[Risk Check] Available dollar amount {dollar_amt:.2f} too small for a SELL order; skipping.")
                else:
                    order_quantity = round(dollar_amt / limit_price, 3)
                    approval = (
                        self.risk_manager.check_available_balance(dollar_amt)
                        and self.risk_manager.check_buy_order_value(limit_price)
                        and self.risk_manager.check_sell_position()
                    )
                print(f"{RED}[Risk Check] {action} approval: {approval}, Amount: {dollar_amt:.2f}, Quantity: {order_quantity}{RESET}")
                logging.info(f"[Risk Check] {action} approval: {approval}, Amount: {dollar_amt:.2f}, Quantity: {order_quantity}")

        if not approval:
            print(f"{YELLOW}Trade not approved for action: {action}{RESET}")
            logging.warning(f"[Risk Check] Trade not approved for action: {action}")
            return

        if not order_queue_ok:
            print(f"{YELLOW}Order queue limit reached; cannot place new orders.{RESET}")
            logging.warning("[Order Management] Order queue limit reached; cannot place new orders.")
            return

        if order_quantity <= 0:
            print(f"{RED}Calculated order quantity invalid or zero ({order_quantity}); aborting order.{RESET}")
            logging.error(f"[Order Placement] Calculated order quantity invalid or zero ({order_quantity}); aborting order.")
            return

        # Construct and place order
        order_data = {
            "symbol": "BTCUSDT",
            # "price": limit_price, # Market orders don't use a price
            "side": direction,
            "type": "MARKET",  # Change "LIMIT" to "MARKET"
            "quantity": order_quantity,
            "timestamp": servertime - OFFSET,
            "recvWindow": 60000,
            # "timeInForce": "GTC", # Not needed for MARKET orders
        }
        print(f"{GREEN}Submitting {direction} order: {order_data}{RESET}")
        logging.info(f"[Order Placement] Submitting {direction} order: {order_data}")
        trade_result = self.trade_executor.execute_trade(order_data, "trade")

        if trade_result:
            print(f"{GREEN}{direction} order executed successfully at {limit_price}.{RESET}")
            logging.info(f"[ExecManager] {direction} order executed successfully at {limit_price}.")
            self.book_keeper.update_bookkeeper(datetime.now(), limit_price, servertime)
            self.book_keeper.return_historical_data().to_csv("historical_data.csv", mode='a', header=not os.path.exists("historical_data.csv"))
            
            # <<<<<<<< PRINT REPORT >>>>>>>>
            self.print_performance_report()
        
        else:
            print(f"{RED}{direction} order placement failed.{RESET}")
            logging.error(f"[ExecManager] {direction} order placement failed.")


    def print_performance_report(self):
        """Calculates and prints a summary of key performance metrics."""

        logging.info("--- PERFORMANCE REPORT ---")

        # Ensure there is enough data to calculate metrics
        if self.book_keeper.historical_data.shape[0] < 2:
            logging.info("Not enough data to generate a report.")
            return

        try:
            # Get the metrics from the book_keeper
            sharpe_ratio = self.book_keeper.calculate_sharpe_ratio()
            max_drawdown = self.book_keeper.calculate_max_drawdown()

            # Get latest P&L figures
            realized_pnl = self.book_keeper.get_realized_pnl
            unrealized_pnl = self.book_keeper.get_unrealized_pnl
            wallet_balance = self.book_keeper.get_wallet_balance
            
            signal, signal_price, signal_timestamp = self.strategy.predict()

            # Print the formatted report
            print(f"{CYAN}--- PERFORMANCE REPORT ---{RESET}")
            print(f"{GREEN}      Wallet Balance: ${wallet_balance:,.2f}{RESET}")
            print(f"{GREEN}    Unrealized P&L: ${unrealized_pnl:,.2f}{RESET}")
            print(f"{GREEN}      Realized P&L: ${realized_pnl:,.2f}{RESET}")
            print(f"{CYAN}      Sharpe Ratio: {sharpe_ratio:.4f}{RESET}")
            print(f"{CYAN}   Maximum Drawdown: {max_drawdown:.2%}{RESET}")
            print(f"{CYAN}        Signal: {signal}{RESET}")
            print(f"{CYAN}      Close: {signal_price}{RESET}")
            print(f"{CYAN}--------------------------{RESET}")

            # Print the formatted report
            logging.info(f"      Wallet Balance: ${wallet_balance:,.2f}")
            logging.info(f"    Unrealized P&L: ${unrealized_pnl:,.2f}")
            logging.info(f"      Realized P&L: ${realized_pnl:,.2f}")
            logging.info(f"      Sharpe Ratio: {sharpe_ratio:.4f}")
            logging.info(f"   Maximum Drawdown: {max_drawdown:.2%}")
            logging.info(f"      Signal: {signal}")
            logging.info(f"      Close: {signal_price}")
            
            # --- Write to CSV ---
            csv_file = "performance_report.csv"
            file_exists = os.path.isfile(csv_file)
            fieldnames = [
                "timestamp", "wallet_balance", "unrealized_pnl",
                "realized_pnl", "sharpe_ratio", "max_drawdown", "Close", "signal"
            ]

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "wallet_balance": wallet_balance,
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": realized_pnl,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "Close": signal_price, 
                    "signal": signal 
                })

        except Exception as e:
            logging.error(f"Could not generate performance report: {e}")

        logging.info("--------------------------")

def on_exec():
    """
    Callback function for trade execution events (optional, can be used for logging or analytics).
    """
    print("Execution callback triggered.")


def heartbeat_loop(exec_manager):
    heartbeat_count = 0
    while True:
        time.sleep(10)
        heartbeat_count += 1
        print("Heartbeat: application running.")
        exec_manager.print_performance_report()

if __name__ == "__main__":
    # --- Load credentials from .env and initialize main system objects ---
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
    
    # --- Run heartbeat in a thread ---
    t = threading.Thread(target=heartbeat_loop, args=(exec_manager,), daemon=True)
    t.start()

    # --- Start live plotting (this MUST be on the main thread for macOS) ---
    live_performance_plot()  # This blocks until you close the plot window
            
    # heartbeat_count = 0
    # while True:
    #     time.sleep(10)
    #     heartbeat_count += 1
    #     print("Heartbeat: application running.")

    #     # Print a performance report every 5 heartbeats (50 seconds)
    #     if heartbeat_count % 1 == 0:
    #         exec_manager.print_performance_report()

    # # After initializing exec_manager and before while True heartbeat
