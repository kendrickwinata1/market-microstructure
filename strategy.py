import logging
import numpy as np

class MarketMakerStrategy:
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.config = config
        self.inventory = 0.0
        self.pnl = 0.0
        self.trade_history = []
        self.last_bid_order = None
        self.last_ask_order = None

    def calculate_quotes(self, mid_price):
        spread = mid_price * self.config.spread_pct / 100.0
        inventory_skew = self.config.inventory_skew_factor * (self.inventory - self.config.inventory_target)
        bid = mid_price - spread/2 - inventory_skew
        ask = mid_price + spread/2 - inventory_skew
        return round(bid, 1), round(ask, 1)

    def run_step(self):
        # 1. Cancel previous orders
        self.exchange.cancel_all_orders()
        # 2. Get latest order book
        bid_px, ask_px = self.exchange.get_order_book()
        mid_px = (bid_px + ask_px) / 2
        # 3. Compute skewed quotes
        bid, ask = self.calculate_quotes(mid_px)
        # 4. Place new limit orders
        result_bid = self.exchange.place_order("BUY", self.config.order_size, bid)
        result_ask = self.exchange.place_order("SELL", self.config.order_size, ask)
        logging.info(f"Quotes: Bid={bid:.2f}, Ask={ask:.2f} (Inventory={self.inventory:.4f})")
        self.last_bid_order, self.last_ask_order = result_bid, result_ask
        # 5. Log balances
        balance = self.exchange.get_balance()
        logging.info(f"USDT balance: {balance}")
        # 6. Risk check (optional: implement stop if drawdown etc.)
        # TODO: update PnL & inventory as orders fill (need fill notification or polling in real system)
        # For live mode, you need to check open orders/fills for updating inventory!
        return True
