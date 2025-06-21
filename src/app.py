import tkinter as tk  # Placeholder for potential GUI features; not used directly here
from queue import Queue
from threading import Thread
from datetime import datetime
from dotenv import load_dotenv
import os
import time
import logging

import tkinter as tk
from trading_engine.main_trading_strategy import TradingStrategy
from visualization.live_plotter import LivePlotter

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

# Optional: ANSI color codes for terminal output (works in most terminals)
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

    # def exec_strat(self, tick):
    #     """
    #     Main event handler for each market tick.
    #     - Updates bookkeeping and cancels stale orders
    #     - Checks risk triggers for liquidation
    #     - Otherwise, calls the trading model, checks risk for order, and places order if approved
    #     """
        
    #     last_price = tick["lastprice"]
    #     if not last_price:
    #         logging.warning("[ExecManager] Received empty last price, skipping tick.")
    #         return

    #     server_response = self.rest_gateway.time()
    #     servertime = int(server_response.get("serverTime", 0))
    #     if not servertime:
    #         logging.error("[ExecManager] Server time fetch failed.")
    #         return

    #     today = datetime.fromtimestamp(servertime / 1000).date()
    #     self.book_keeper.update_bookkeeper(today, last_price, servertime)

    #     # Order management
    #     open_orders = self.rest_gateway.get_all_open_orders("BTCUSDT", servertime)
    #     order_queue_ok = len(open_orders) < MAX_OPEN_ORDER_COUNT

    #     # TO DEBUG
    #     # Risk checks 
    #     if self.risk_manager.trigger_stop_loss() or self.risk_manager.trigger_trading_halt():
    #         logging.info("[ExecManager] Risk trigger activated. Attempting liquidation.")
    #         self.handle_liquidation(servertime)
    #         return

    #     # Strategy analysis
    #     self.update_queue(tick)
    #     self.strategy.collect_new_data()
    #     self.strategy.aggregate_data()
    #     model_output = self.strategy.analyze_data()

    #     if not model_output:
    #         self.model_none_count += 1
    #         if self.model_none_count >= MAX_MODEL_NONE_COUNT:
    #             print("Model returned None too many times; cancelling all orders.")
    #             self.rest_gateway.cancel_all_order("BTCUSDT", servertime)
    #         print(f"MODEL NONE COUNT VALUE = {self.model_none_count}")
    #         return

    #     direction, limit_price = model_output[0].upper(), float(model_output[1])

    #     # --- Override HOLD based on momentum indicator ---
    #     # if direction == "HOLD":
    #     #     try:
    #     #         data = self.strategy.data
    #     #         last_deriv = data["Short_Moving_Avg_1st_Deriv"].iloc[-1]
    #     #         if last_deriv > 10:
    #     #             direction = "BUY"
    #     #             print("[Override] HOLD -> BUY due to momentum")
    #     #         elif last_deriv < -10:
    #     #             direction = "SELL"
    #     #             print("[Override] HOLD -> SELL due to momentum")
    #     #     except Exception as e:
    #     #         print("[Override Error] Cannot override HOLD:", e)
        
    #     if direction == "HOLD":
    #         # Force override HOLD signal for testing
    #         direction = "BUY"  # or "SELL" to test short-side logic
    #         print("[Force Override] HOLD signal forcibly overridden to BUY for testing")


    #     # Prepare order details
    #     order_quantity = 0
    #     approval = False

    #     if direction == "BUY":
    #         dollar_amt = self.risk_manager.get_available_tradable_balance()
    #         order_quantity = 0.001
    #         # order_quantity = max(round(dollar_amt / limit_price, 3), 0.001)  # Force min lot size
    #         # order_quantity = round(dollar_amt / limit_price, 3)
    #         # approval = (
    #         #     self.risk_manager.check_available_balance(dollar_amt)
    #         #     and self.risk_manager.check_buy_order_value(limit_price)
    #         #     and self.risk_manager.check_buy_position()
    #         # )
            
    #         # Force approval for testing
    #         approval = True
    #         print("[Force Override] Risk manager approval forced to True for BUY.")

    #     elif direction == "SELL":
    #         pos_info = self.rest_gateway.get_position_info("BTCUSDT", servertime)
    #         order_quantity = float(pos_info[0]["positionAmt"]) if pos_info else 0
    #         # approval = (
    #         #     self.risk_manager.check_short_position(order_quantity)
    #         #     and self.risk_manager.check_sell_order_value(limit_price)
    #         # )

    #         # Force approval for testing
    #         approval = True
    #         print("[Force Override] Risk manager approval forced to True for SELL.")

    #     elif direction == "HOLD":
    #         print("Model signals HOLD.")
    #         return  # nothing else to do

    #     # --- Final order placement if all checks pass ---
    #     if approval and order_queue_ok and order_quantity > 0:
    #         order_data = {
    #             "symbol": "BTCUSDT",
    #             "price": limit_price,
    #             "side": direction,
    #             "type": "LIMIT",
    #             "quantity": order_quantity,
    #             "timestamp": servertime - OFFSET,
    #             "recvWindow": 60000,
    #             "timeInForce": "GTC",
    #         }
    #         trade_result = self.trade_executor.execute_trade(order_data, "trade")

    #         if trade_result:
    #             logging.info(f"[ExecManager] {direction} order placed successfully.")
    #             self.book_keeper.update_bookkeeper(datetime.now(), limit_price, servertime) 
    #             self.book_keeper.return_historical_data().to_csv("historical_data.csv")
    #         else:
    #             logging.error(f"[ExecManager] {direction} order placement failed.")
    #     else:
    #         print("Order not approved by risk manager or order queue full.")

    # def exec_strat(self, tick):
    #     """
    #     Main event handler for each market tick. 
    #     Logic:
    #     1. Update bookkeeper.
    #     2. Check for old open orders and cancel if stale.
    #     3. Risk check for liquidation (stop loss or trading halt).
    #     4. If not liquidating, run model and, if approved, place order.
    #     5. Handle model None returns and excessive None counts.
    #     """

    #     last_price = tick["lastprice"]
    #     print(f"what is S even {last_price}")

    #     if last_price != "":
    #         # 1. Get server time and date
    #         server_response = self.rest_gateway.time()
    #         servertime = int(server_response.get("serverTime", 0))
    #         servertime_dt = datetime.fromtimestamp(servertime / 1000)
    #         the_date = servertime_dt.date()

    #         # 2. Update bookkeeper
    #         self.book_keeper.update_bookkeeper(the_date, last_price, servertime)

    #         # 3. Check for old open orders and cancel if needed
    #         print("XXXXXXXXXXXXXXXXXX   FIRST, LET US CHECK POSITION    XXXXXXXXXXXXXXXXXX")
    #         current_open_orders = self.rest_gateway.get_all_open_orders("BTCUSDT", servertime)
    #         print(current_open_orders)
    #         order_queue_ok = True

    #         if len(current_open_orders) >= MAX_OPEN_ORDER_COUNT:
    #             for x in current_open_orders:
    #                 x_dt = datetime.fromtimestamp(x["time"] / 1000)
    #                 timediff = servertime_dt - x_dt
    #                 timediff_seconds = timediff.total_seconds()
    #                 print(f"the time diff is {timediff_seconds}")
    #                 if timediff_seconds > MAX_OPEN_ORDER_LIFE_SECONDS:
    #                     print("CANCELLING ORDERS")
    #                     self.rest_gateway.cancel_order("BTCUSDT", servertime, x["orderId"])
    #                     order_queue_ok = True
    #                 else:
    #                     print("NO CANCELLABLE ORDERS")
    #                     order_queue_ok = False
    #         else:
    #             order_queue_ok = True

    #         # 4. Risk triggers (liquidation logic)
    #         stop_loss_trigger = self.risk_manager.trigger_stop_loss()
    #         trading_halt_trigger = self.risk_manager.trigger_trading_halt()
    #         print(f"stop_loss_trig {stop_loss_trigger} ; trading_halt_trig {trading_halt_trigger}")
    #         liquidate_approval = stop_loss_trigger or trading_halt_trigger

    #         print(f"LIQUIDATE CHECK : {liquidate_approval} OR {self.reattempt_liquidate}")

    #         if liquidate_approval or self.reattempt_liquidate:
    #             current_position_resp = self.rest_gateway.get_position_info("BTCUSDT", servertime)
    #             print("we will be liquidating all")
    #             if server_response is not None:
    #                 self.reattempt_liquidate = False
    #                 servertime = int(server_response.get("serverTime", 0))
    #                 # Cancel all open orders
    #                 cancel_resp = self.rest_gateway.cancel_all_order("BTCUSDT", servertime)
    #                 print(cancel_resp)
    #                 # Close all positions
    #                 current_position_resp = self.rest_gateway.get_position_info("BTCUSDT", servertime)
    #                 if current_position_resp is not None:
    #                     position_amt = float(current_position_resp[0]["positionAmt"])
    #                     if position_amt > 0:
    #                         liquidate_data = {
    #                             "symbol": "BTCUSDT",
    #                             "side": "SELL",
    #                             "type": "MARKET",
    #                             "quantity": position_amt,
    #                             "timestamp": servertime - OFFSET,
    #                             "recvWindow": 60000,
    #                         }
    #                         print(liquidate_data)
    #                         self.trade_executor.execute_trade(liquidate_data, "trade")
    #                         print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    #                     else:
    #                         print("NO POSITION TO LIQUIDATE")
    #                         print("xxxxxxxxxxxxxxxxxxxxxxxxx NO POSITION TO LIQUIDATE xxxxxxxxxxxxxxxxxxxxxxxxx")
    #             else:
    #                 print("cannot get response from server !")
    #                 self.reattempt_liquidate = True

    #         # 5. Proceed with normal trading if not liquidating
    #         else:
    #             self.update_queue(tick)
    #             self.strategy.collect_new_data()
    #             self.strategy.aggregate_data()
    #             model_output = self.strategy.analyze_data()
    #             print("model output: ", model_output)

    #             if model_output is not None:
    #                 direction = model_output[0].upper()
    #                 limit_price = float(model_output[1])

    #                 # Get server time before placing order
    #                 server_response = self.rest_gateway.time()
    #                 servertime = int(server_response.get("serverTime", 0))

    #                 order_quantity = 0
    #                 approval = False

    #                 if direction == "BUY":
    #                     dollar_amt_buy = self.risk_manager.get_available_tradable_balance()
    #                     order_quantity = round(dollar_amt_buy / limit_price, 3)
    #                     buy_balance_check = self.risk_manager.check_available_balance(dollar_amt_buy)
    #                     buy_price_check = self.risk_manager.check_buy_order_value(limit_price)
    #                     buy_position_check = self.risk_manager.check_buy_position()
    #                     print(f"buy_balance_check: {buy_balance_check} ,buy_price_check : {buy_price_check}, buy_position_check: {buy_position_check}")
    #                     approval = buy_balance_check and buy_price_check and buy_position_check

    #                 elif direction == "SELL":
    #                     current_position_resp = self.rest_gateway.get_position_info("BTCUSDT", servertime)
    #                     order_quantity = float(current_position_resp[0]["positionAmt"]) if current_position_resp else 0
    #                     short_pos_check = self.risk_manager.check_short_position(order_quantity)
    #                     sell_price_check = self.risk_manager.check_sell_order_value(limit_price)
    #                     sell_position_check = self.risk_manager.check_sell_position()
    #                     print(f"short pos check: {short_pos_check} , sell_price_check : {sell_price_check}, sell_position_check: {sell_position_check}")
    #                     approval = short_pos_check and sell_price_check and sell_position_check

    #                 elif direction == "HOLD":
    #                     approval = False
    #                     print("MODEL SIGNALS HOLD")
    #                 else:
    #                     approval = False
    #                     print("invalid direction")

    #                 print(f"{direction} --> ORDER QUANTITY {order_quantity}, approved ? {approval} and order queue {order_queue_ok}")

    #                 if approval and order_queue_ok and order_quantity > 0:
    #                     if len(current_open_orders) < MAX_OPEN_ORDER_COUNT:
    #                         self.model_none_count = 0
    #                         order_data = {
    #                             "symbol": "BTCUSDT",
    #                             "price": limit_price,
    #                             "side": direction,
    #                             "type": "LIMIT",
    #                             "quantity": order_quantity,
    #                             "timestamp": servertime - OFFSET,
    #                             "recvWindow": 60000,
    #                             "timeInForce": "GTC",
    #                         }
    #                         print(order_data)
    #                         self.trade_executor.execute_trade(order_data, "trade")
    #                         print("my limit price: ", limit_price)
    #                         self.book_keeper.update_bookkeeper(datetime.now(), limit_price, servertime)
    #                         get_pnl = self.book_keeper.return_historical_data()
    #                         print(f"check historical position : {self.book_keeper.historical_positions.tail(3)}")
    #                         get_pnl.to_csv("historical_data.csv")
    #                 else:
    #                     print("SORRY CANT TRADE")
    #             else:
    #                 self.model_none_count += 1
    #                 if self.model_none_count >= MAX_MODEL_NONE_COUNT:
    #                     print("possible model error? Cancel All orders")
    #                     cancel_resp = self.rest_gateway.cancel_all_order("BTCUSDT", servertime)
    #                     print(f"CANCEL MODEL NONE: {cancel_resp}")
    #                 print(f"MODEL NONE COUNT VALUE = {self.model_none_count}")

    # def exec_strat(self, tick):
    #     last_price = tick["lastprice"]
    #     print(f"\n{CYAN}{'='*18} NEW TICK {'='*18}{RESET}")
    #     print(f"{YELLOW}Tick received at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Last price: {last_price}{RESET}")

    #     if last_price != "":
    #         server_response = self.rest_gateway.time()
    #         servertime = int(server_response.get("serverTime", 0))
    #         servertime_dt = datetime.fromtimestamp(servertime / 1000)
    #         the_date = servertime_dt.date()

    #         # 2. Bookkeeper update
    #         print(f"{CYAN}Updating bookkeeper for date {the_date}, price {last_price}...{RESET}")
    #         self.book_keeper.update_bookkeeper(the_date, last_price, servertime)

    #         # 3. Open order management
    #         print(f"{CYAN}Checking open orders...{RESET}")
    #         current_open_orders = self.rest_gateway.get_all_open_orders("BTCUSDT", servertime)
    #         print(f"{CYAN}Open orders:{RESET} {current_open_orders if current_open_orders else '[None]'}")
    #         order_queue_ok = True

    #         for x in current_open_orders:
    #             x_dt = datetime.fromtimestamp(x["time"] / 1000)
    #             timediff = servertime_dt - x_dt
    #             timediff_seconds = timediff.total_seconds()
    #             print(f"{YELLOW}Order {x['orderId']} open for {timediff_seconds:.2f}s{RESET}")
    #             if timediff_seconds > MAX_OPEN_ORDER_LIFE_SECONDS:
    #                 print(f"{RED}Cancelling stale order {x['orderId']}{RESET}")
    #                 self.rest_gateway.cancel_order("BTCUSDT", servertime, x["orderId"])
    #                 order_queue_ok = True
    #             else:
    #                 order_queue_ok = False

    #         # 4. Risk trigger check
    #         stop_loss_trigger = self.risk_manager.trigger_stop_loss()
    #         trading_halt_trigger = self.risk_manager.trigger_trading_halt()
    #         print(f"{CYAN}Risk check:{RESET} stop_loss: {stop_loss_trigger}, trading_halt: {trading_halt_trigger}")
    #         liquidate_approval = stop_loss_trigger or trading_halt_trigger

    #         # Liquidation logic
    #         if liquidate_approval or self.reattempt_liquidate:
    #             print(f"{RED}LIQUIDATION TRIGGERED! Attempting to close all positions...{RESET}")
    #             # ... existing liquidation logic ...
    #         else:
    #             # 5. Normal strategy flow
    #             self.update_queue(tick)
    #             self.strategy.collect_new_data()
    #             self.strategy.aggregate_data()
    #             model_output = self.strategy.analyze_data()
    #             print(f"{CYAN}Model output:{RESET} {model_output}")
                
    #             order_quantity = 0
    #             approval = False  # set a default at the start
                
    #             if model_output is not None:
    #                 direction = model_output[0].upper()
    #                 limit_price = float(model_output[1])
    #                 print(f"{GREEN}Signal: {direction} | Limit price: {limit_price}{RESET}")

    #                 # Check and print risk results for buy/sell
    #                 if direction == "BUY":
    #                     # ...risk check logic...
    #                     print(f"{GREEN}Buy risk checks passed: {approval}{RESET}")
    #                 elif direction == "SELL":
    #                     # ...risk check logic...
    #                     print(f"{RED}Sell risk checks passed: {approval}{RESET}")
    #                 elif direction == "HOLD":
    #                     print(f"{YELLOW}Signal is HOLD. No trade executed.{RESET}")

    #                 print(f"{CYAN}Order approved? {approval} | Order queue ok? {order_queue_ok} | Quantity: {order_quantity}{RESET}")

    #                 if approval and order_queue_ok and order_quantity > 0:
    #                     print(f"{GREEN}Placing {direction} LIMIT order for {order_quantity} BTCUSDT at {limit_price}{RESET}")
    #                     # ...order placement logic...
    #                 else:
    #                     print(f"{YELLOW}Trade not placed: Approval or order queue conditions not met.{RESET}")
    #             else:
    #                 print(f"{YELLOW}Model returned None. Skipping this tick.{RESET}")

    #         print(f"{CYAN}{'='*50}{RESET}\n")
    
    def exec_strat(self, tick):
        # Terminal color codes
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        offset = 15000  # timestamp offset

        last_price = tick["lastprice"]
        print(f"\n{CYAN}{'='*18} NEW TICK {'='*18}{RESET}")
        print(f"{YELLOW}Tick received at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Last price: {last_price}{RESET}")

        if last_price != "":
            # 1. GET ENTER EXEC STRAT TIME
            response = self.rest_gateway.time()
            servertime = int(response.get("serverTime", 0))
            servertime_dt = datetime.fromtimestamp(servertime / 1000)
            the_date = servertime_dt.date()

            # 2. UPDATE THE BOOK KEEPER
            print(f"{CYAN}Updating bookkeeper for date {the_date}, price {last_price}...{RESET}")
            self.book_keeper.update_bookkeeper(the_date, last_price, servertime)

            # 3. CHECK ANY OLDER ORDERS NEED TO CANCEL OR NOT
            print(f"{CYAN}Checking open orders...{RESET}")
            current_open_orders = self.rest_gateway.get_all_open_orders("BTCUSDT", servertime)
            print(f"{CYAN}Open orders:{RESET} {current_open_orders if current_open_orders else '[None]'}")
            order_queue_ok = True

            # if len(current_open_orders) >= MAX_OPEN_ORDER_COUNT:
            #     for x in current_open_orders:
            #         servertime_dt = datetime.fromtimestamp(servertime / 1000)
            #         x_dt = datetime.fromtimestamp(x["time"] / 1000)
            #         timediff = servertime_dt - x_dt
            #         timediff_seconds = timediff.total_seconds()
            #         print(f"{YELLOW}Order {x['orderId']} open for {timediff_seconds:.2f}s{RESET}")
            #         if timediff_seconds > MAX_OPEN_ORDER_LIFE_SECONDS:
            #             print(f"{RED}Cancelling stale order {x['orderId']}{RESET}")
            #             self.rest_gateway.cancel_order("BTCUSDT", servertime, x["orderId"])
            #             order_queue_ok = True
            #         else:
            #             print(f"{YELLOW}No cancellable orders.{RESET}")
            #             order_queue_ok = False
            # else:
            #     order_queue_ok = True
            

            # TRY WITH OTHER LOGIC
            
            if len(current_open_orders) >= MAX_OPEN_ORDER_COUNT:
                for x in current_open_orders:
                    servertime_dt = datetime.fromtimestamp(servertime / 1000)
                    x_dt = datetime.fromtimestamp(x["time"] / 1000)
                    timediff = servertime_dt - x_dt
                    timediff_seconds = timediff.total_seconds()
                    if timediff_seconds > 3:   # Try 3 seconds, or even lower
                        print("CANCELLING ORDERS")
                        self.rest_gateway.cancel_order("BTCUSDT", servertime, x["orderId"])
                        order_queue_ok = True
                    else:
                        # If too soon, still allow new orders
                        order_queue_ok = True
            else:
                order_queue_ok = True

            # 4. LIQUIDATE CHECK
            stop_loss_trigger = self.risk_manager.trigger_stop_loss()
            trading_halt_trigger = self.risk_manager.trigger_trading_halt()
            print(f"{CYAN}Risk check:{RESET} stop_loss: {stop_loss_trigger}, trading_halt: {trading_halt_trigger}")
            liquidate_approval = stop_loss_trigger or trading_halt_trigger
            print(f"{CYAN}LIQUIDATE CHECK : {liquidate_approval} OR {self.reattempt_liquidate}{RESET}")

            # LIQUIDATION LOGIC
            if liquidate_approval or self.reattempt_liquidate:
                current_position_resp = self.rest_gateway.get_position_info("BTCUSDT", servertime)
                print(f"{RED}LIQUIDATION TRIGGERED! Attempting to close all positions...{RESET}")
                if response is not None:
                    self.reattempt_liquidate = False
                    servertime = int(response["serverTime"])

                    # 1. CANCEL ALL STANDING ORDERS
                    cancel_resp = self.rest_gateway.cancel_all_order("BTCUSDT", servertime)
                    print(f"{YELLOW}Cancel all orders response: {cancel_resp}{RESET}")

                    # 2. CLOSE ALL POSITIONS
                    current_position_resp = self.rest_gateway.get_position_info("BTCUSDT", servertime)
                    if current_position_resp is not None:
                        position_amt = float(current_position_resp[0]["positionAmt"])
                        if position_amt > 0:
                            liquidate_data = {
                                "symbol": "BTCUSDT",
                                "side": "SELL",
                                "type": "MARKET",
                                "quantity": position_amt,
                                "timestamp": servertime - offset,
                                "recvWindow": 60000,
                            }
                            print(f"{RED}Executing liquidation: {liquidate_data}{RESET}")
                            self.trade_executor.execute_trade(liquidate_data, "trade")
                            print(f"{RED}{'x'*40} LIQUIDATION DONE {'x'*40}{RESET}")
                        else:
                            print(f"{YELLOW}NO POSITION TO LIQUIDATE{RESET}")
                    else:
                        print(f"{RED}Cannot get position info for liquidation.{RESET}")
                else:
                    print(f"{RED}Cannot get server time, will retry liquidation!{RESET}")
                    self.reattempt_liquidate = True

            # 5. PROCEED AS NORMAL IF NOT LIQUIDATED
            else:
                self.update_queue(tick)
                self.strategy.collect_new_data()
                self.strategy.aggregate_data()
                model_output = self.strategy.analyze_data()
                # model_output = self.strategy.analyze_data(self.current_position)
                print(f"{CYAN}Model output:{RESET} {model_output}")

                order_quantity = 0
                approval = False

                if model_output is not None:
                    direction = model_output[0].upper()
                    limit_price = float(model_output[1])
                    print(f"{GREEN}Signal: {direction} | Limit price: {limit_price}{RESET}")

                    response = self.rest_gateway.time()
                    servertime = int(response.get("serverTime", 0))

                    if direction == "BUY":
                        dollar_amt_buy = self.risk_manager.get_available_tradable_balance()
                        order_quantity = round(dollar_amt_buy / limit_price, 3)
                        buy_balance_check = self.risk_manager.check_available_balance(dollar_amt_buy)
                        buy_price_check = self.risk_manager.check_buy_order_value(limit_price)
                        buy_position_check = self.risk_manager.check_buy_position()
                        print(f"{GREEN}buy_balance_check: {buy_balance_check}, buy_price_check: {buy_price_check}, buy_position_check: {buy_position_check}{RESET}")
                        approval = buy_balance_check and buy_price_check and buy_position_check

                    elif direction == "SELL":
                        current_position_resp = self.rest_gateway.get_position_info("BTCUSDT", servertime)
                        order_quantity = float(current_position_resp[0]["positionAmt"]) if current_position_resp else 0
                        short_pos_check = self.risk_manager.check_short_position(order_quantity)
                        sell_price_check = self.risk_manager.check_sell_order_value(limit_price)
                        sell_position_check = self.risk_manager.check_sell_position()
                        print(f"{RED}short_pos_check: {short_pos_check}, sell_price_check: {sell_price_check}, sell_position_check: {sell_position_check}{RESET}")
                        approval = short_pos_check and sell_price_check and sell_position_check

                    elif direction == "HOLD":
                        approval = 0
                        print(f"{YELLOW}MODEL SIGNALS HOLD{RESET}")

                    else:
                        approval = 0
                        print(f"{RED}Invalid direction: {direction}{RESET}")

                    print(f"{CYAN}{direction} --> ORDER QUANTITY {order_quantity}, approved? {approval} and order queue {order_queue_ok}{RESET}")

                    if approval and order_queue_ok and order_quantity > 0:
                        
                        # Decide limit price to use
                        # Optionally, for SELL use just below market, for BUY just above
                        if direction == "SELL":
                            my_limit_price = float(last_price) * 0.999
                        elif direction == "BUY":
                            my_limit_price = float(last_price) * 1.001
                        else:
                            my_limit_price = float(last_price)
                            

                        print(f"{GREEN}Placing {direction} LIMIT order for {order_quantity} BTCUSDT at {limit_price}{RESET}")
                        order_data = {
                            "symbol": "BTCUSDT",
                            "price": my_limit_price,
                            "side": direction,
                            "type": "LIMIT",
                            "quantity": order_quantity,
                            "timestamp": servertime - offset,
                            "recvWindow": 60000,
                            "timeInForce": "GTC",
                        }
                        result = self.trade_executor.execute_trade(order_data, "trade")
                        print(f"{CYAN}Order placement result:{RESET} {result}")

                        # Bookkeeper update after trade
                        self.book_keeper.update_bookkeeper(datetime.now(), limit_price, servertime)
                        pnl = self.book_keeper.return_historical_data()
                        print(f"{CYAN}Check historical positions (last 3):\n{self.book_keeper.historical_positions.tail(3)}{RESET}")
                        pnl.to_csv("historical_data.csv")
                    else:
                        print(f"{YELLOW}SORRY CANT TRADE{RESET}")
                else:
                    self.model_none_count += 1
                    if self.model_none_count >= MAX_MODEL_NONE_COUNT:
                        print(f"{RED}Model returned None too many times. Cancelling all orders!{RESET}")
                        cancel_resp = self.rest_gateway.cancel_all_order("BTCUSDT", servertime)
                        print(f"{RED}CANCEL MODEL NONE: {cancel_resp}{RESET}")
                    print(f"{YELLOW}MODEL NONE COUNT VALUE = {self.model_none_count}{RESET}")

        print(f"{CYAN}{'='*50}{RESET}\n")


def on_exec():
    """
    Callback function for trade execution events (optional, can be used for logging or analytics).
    """
    print("Execution callback triggered.")
    
# TEST
def trading_bot_main(strategy, exec_manager, data_stream):
    """
    The trading bot logic runs in a background thread.
    """
    data_stream.register_tick_callback(exec_manager.exec_strat)
    data_stream.connect()
    # Main bot heartbeat loop
    while True:
        time.sleep(10)
        print("Heartbeat: application running.")
        


if __name__ == "__main__":
    # --- Load credentials from ..env and initialize main system objects ---
    
    shared_queue = Queue()
    strategy = TradingStrategy(shared_queue)
    
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
   
 #TRY
    exec_manager.strategy = strategy
    exec_manager.queue = shared_queue

    # --- Market data stream setup: every tick triggers trading logic in exec_manager ---
    data_stream = DataStream(symbol, api_key, api_secret)
    data_stream.register_tick_callback(exec_manager.exec_strat)
    data_stream.connect()


    # TRY
    # 1. Start the trading bot in a **background thread**
    bot_thread = Thread(target=trading_bot_main, args=(strategy, exec_manager, data_stream))
    bot_thread.daemon = True
    bot_thread.start()

    # 2. Start the Tkinter GUI (main thread, for Mac compatibility)
    root = tk.Tk()
    plotter = LivePlotter(root, strategy, data_window=60)
    root.mainloop()

    # # --- Main application heartbeat to keep the process alive and provide basic monitoring ---
    # while True:
    #     time.sleep(10)
    #     print("Heartbeat: application running.")


