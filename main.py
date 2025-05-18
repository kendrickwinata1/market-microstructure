# main.py
import logging
from config import Config
from exchange import BinanceFuturesExchange
from strategy import MarketMakerStrategy
from risk_manager import RiskManager
import matplotlib.pyplot as plt

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def run_backtest(config):
    import numpy as np

    # Generate price series (random walk or from file)
    if config.use_random_walk:
        np.random.seed(42)
        price_series = 30000 + np.cumsum(np.random.normal(0, 50, 500))
    else:
        import pandas as pd
        price_series = pd.read_csv(config.price_series_file)['price'].values

    # Simple simulator
    class SimulatedExchange:
        def __init__(self, price_series):
            self.price_series = price_series
            self.idx = 0
            self.inventory = 0.0
            self.balance = 10000.0  # Start with $10k USDT

        def get_order_book(self, limit=5):
            px = self.price_series[self.idx]
            return px-1, px+1

        def place_order(self, side, quantity, price, reduce_only=False):
            fill_px = self.price_series[self.idx]
            if (side == "BUY" and fill_px <= price) or (side == "SELL" and fill_px >= price):
                if side == "BUY":
                    self.inventory += quantity
                    self.balance -= quantity * price
                else:
                    self.inventory -= quantity
                    self.balance += quantity * price
                return {"status": "FILLED", "side": side, "price": price, "qty": quantity}
            else:
                return {"status": "NEW", "side": side, "price": price, "qty": quantity}

        def cancel_all_orders(self):
            return None

        def get_balance(self, asset="USDT"):
            return self.balance

    exchange = SimulatedExchange(price_series)
    risk_manager = RiskManager(config)
    strategy = MarketMakerStrategy(exchange, config, risk_manager)

    pnl_curve = []
    inventory_curve = []
    price_curve = []

    for i in range(len(price_series)):
        exchange.idx = i
        strategy.run_step()
        price_curve.append(price_series[i])
        inventory_curve.append(exchange.inventory)
        pnl_curve.append(exchange.balance + exchange.inventory * price_series[i])

    print(f"Backtest complete. Final PnL: {pnl_curve[-1]:.2f}")

    # Plot results
    if config.enable_plotting:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(price_curve, label='Price')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(inventory_curve, label='Inventory')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(pnl_curve, label='PnL')
        plt.legend()
        plt.tight_layout()
        plt.show()

def run_live(config):
    exchange = BinanceFuturesExchange(config)
    risk_manager = RiskManager(config)
    strategy = MarketMakerStrategy(exchange, config, risk_manager)
    import time
    while True:
        strategy.run_step()
        time.sleep(config.refresh_interval)

if __name__ == "__main__":
    config = Config()
    setup_logger(config.log_file)
    if config.mode == "backtest":
        run_backtest(config)
    elif config.mode == "live":
        run_live(config)
    else:
        print("Unknown mode in config.")
