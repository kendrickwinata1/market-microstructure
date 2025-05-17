# strategy.py
import logging
import time
import numpy as np

class SimulatedExchange:
    def __init__(self, config, price_series):
        self.symbol = config.symbol
        self.data = price_series
        self.current_index = 0
        self.current_price = None
        self.current_bid = None
        self.current_ask = None
        self.open_orders = []
        self.inventory = config.initial_inventory
        self.balance = config.initial_balance
        self.starting_balance = config.initial_balance
        self.trade_history = []

    def _update_price(self):
        if self.current_index >= len(self.data):
            return False
        mid = float(self.data[self.current_index])
        self.current_price = mid
        self.current_bid = mid * (1 - 0.0001)
        self.current_ask = mid * (1 + 0.0001)
        self.current_index += 1
        return True

    def get_fair_price(self):
        return self.current_price

    def place_order(self, side, price, quantity):
        order = {"side": side, "price": price, "quantity": quantity, "filled": False}
        if side == "BUY" and price >= self.current_ask:
            fill_price = self.current_ask
            self._execute_fill(order, fill_price)
        elif side == "SELL" and price <= self.current_bid:
            fill_price = self.current_bid
            self._execute_fill(order, fill_price)
        else:
            self.open_orders.append(order)
            logging.info("Simulated %s order: %.4f BTC @ %.2f", side, quantity, price)
        return order

    def _execute_fill(self, order, fill_price):
        side = order["side"]
        qty = order["quantity"]
        order["filled"] = True
        order["fill_price"] = fill_price
        if side == "BUY":
            cost = fill_price * qty
            self.balance -= cost
            self.inventory += qty
        elif side == "SELL":
            revenue = fill_price * qty
            self.balance += revenue
            self.inventory -= qty
        self.trade_history.append({
            "side": side,
            "price": fill_price,
            "quantity": qty,
            "inventory_after": self.inventory,
            "balance_after": self.balance
        })
        logging.info("Sim fill: %s %.4f BTC @ %.2f. Inventory=%.4f, Balance=%.2f",
            side, qty, fill_price, self.inventory, self.balance)

    def cancel_all_orders(self):
        num = len(self.open_orders)
        if num > 0:
            self.open_orders.clear()
            logging.info("Canceled %d simulated open orders", num)

    def step(self):
        if not self._update_price():
            return False
        remaining_orders = []
        for order in self.open_orders:
            if order["side"] == "BUY" and order["price"] >= self.current_ask:
                self._execute_fill(order, fill_price=self.current_ask)
            elif order["side"] == "SELL" and order["price"] <= self.current_bid:
                self._execute_fill(order, fill_price=self.current_bid)
            else:
                remaining_orders.append(order)
        self.open_orders = remaining_orders
        return True

class MarketMakerStrategy:
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.cfg = config
        self.inventory = 0.0
        self.quote_balance = 0.0
        self.pnl = 0.0
        if hasattr(exchange, "inventory"):
            self.inventory = exchange.inventory
            self.quote_balance = exchange.balance

    def calculate_quotes(self):
        fair_price = self.exchange.get_fair_price()
        if fair_price is None:
            return None, None
        base_spread = fair_price * self.cfg.spread_pct
        skew = self.cfg.inventory_skew_factor * (self.inventory - self.cfg.inventory_target) * (fair_price * 0.001)
        adjusted_mid = fair_price + skew
        bid_price = adjusted_mid * (1 - self.cfg.spread_pct / 2)
        ask_price = adjusted_mid * (1 + self.cfg.spread_pct / 2)
        if bid_price >= ask_price:
            bid_price = adjusted_mid * (1 - self.cfg.spread_pct)
            ask_price = adjusted_mid * (1 + self.cfg.spread_pct)
        return bid_price, ask_price

    def check_risk_limits(self, bid_price, ask_price):
        size = self.cfg.order_size
        bid_allowed = True
        ask_allowed = True
        if self.inventory >= self.cfg.position_limit:
            bid_allowed = False
        if self.inventory <= -self.cfg.position_limit:
            ask_allowed = False
        max_buy_qty = max(0.0, self.cfg.position_limit - self.inventory)
        max_sell_qty = max(0.0, self.cfg.position_limit + self.inventory)
        adj_buy_size = min(size, max_buy_qty)
        adj_sell_size = min(size, max_sell_qty)
        return bid_allowed, ask_allowed, adj_buy_size, adj_sell_size

    def update_pnl(self):
        if hasattr(self.exchange, 'balance'):
            current_price = self.exchange.get_fair_price()
            if current_price is None:
                return
            total_value = self.exchange.balance + self.exchange.inventory * current_price
            self.pnl = total_value - self.exchange.starting_balance
        if self.pnl <= -self.cfg.loss_threshold:
            logging.critical("Loss threshold exceeded! PnL=%.2f, stopping trading.", self.pnl)
            try:
                self.exchange.cancel_all_orders()
            except Exception as e:
                logging.error("Error while canceling orders during kill switch: %s", e)
            return False
        return True

    def run_step(self):
        bid_price, ask_price = self.calculate_quotes()
        if bid_price is None or ask_price is None:
            return False
        bid_allowed, ask_allowed, buy_size, sell_size = self.check_risk_limits(bid_price, ask_price)
        self.exchange.cancel_all_orders()
        if bid_allowed and buy_size > 0:
            self.exchange.place_order("BUY", price=bid_price, quantity=buy_size)
        if ask_allowed and sell_size > 0:
            self.exchange.place_order("SELL", price=ask_price, quantity=sell_size)
        logging.info("Quotes: Bid=%.2f, Ask=%.2f (Inventory=%.4f)", bid_price, ask_price, self.inventory)
        if hasattr(self.exchange, "step"):
            self.exchange.step()
            self.inventory = self.exchange.inventory
            self.quote_balance = self.exchange.balance
        return self.update_pnl()
