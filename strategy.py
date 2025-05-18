# strategy.py
import logging

class BaseStrategy:
    def __init__(self, exchange, config, risk_manager):
        self.exchange = exchange
        self.config = config
        self.risk_manager = risk_manager
        self.inventory = 0.0
        self.pnl = 0.0
        self.trade_history = []

    def run_step(self):
        raise NotImplementedError

class MarketMakerStrategy(BaseStrategy):
    def calculate_quotes(self, mid_price):
        spread = mid_price * self.config.spread_pct / 100.0
        inventory_skew = self.config.inventory_skew_factor * (self.inventory - self.config.inventory_target)
        bid = mid_price - spread / 2 - inventory_skew
        ask = mid_price + spread / 2 - inventory_skew
        return round(bid, 1), round(ask, 1)

    def run_step(self):
        # 1. Cancel previous orders
        self.exchange.cancel_all_orders()

        # 2. Get market mid-price
        bid_px, ask_px = self.exchange.get_order_book()
        mid_px = (bid_px + ask_px) / 2

        # 3. Compute skewed quotes
        bid, ask = self.calculate_quotes(mid_px)

        # 4. Risk check BEFORE placing orders
        if not self.risk_manager.check(self.inventory, self.pnl):
            logging.warning("Order skipped due to risk check.")
            return False

        # 5. Place new limit orders
        result_bid = self.exchange.place_order("BUY", self.config.order_size, bid)
        result_ask = self.exchange.place_order("SELL", self.config.order_size, ask)
        logging.info(f"Quotes: Bid={bid:.2f}, Ask={ask:.2f} (Inventory={self.inventory:.4f})")
        self.last_bid_order, self.last_ask_order = result_bid, result_ask

        # 6. Update inventory & pnl if orders filled (for backtest, live: poll/stream fills)
        # NOTE: In real/futures trading, you need to check fill events for precise inventory!
        # Here, update only if orders fill (simulator will handle)
        # For live: add logic to update inventory/pnl from order/fill reports

        # 7. Log balances
        balance = self.exchange.get_balance()
        logging.info(f"USDT balance: {balance}")

        return True
