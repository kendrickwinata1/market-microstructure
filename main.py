# main.py
import logging
import time
from config import Config
from exchange import BinanceTestnetExchange
from strategy import MarketMakerStrategy, SimulatedExchange

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("market_maker.log", mode="w")]
)

def run_live():
    config = Config()
    exchange = BinanceTestnetExchange(config)
    strategy = MarketMakerStrategy(exchange, config)
    logging.info("Starting live trading loop...")
    try:
        while True:
            cont = strategy.run_step()

            # --- Add these lines below to check and print balances ---
            btc_balance = exchange.get_inventory('BTC')
            usdt_balance = exchange.get_inventory('USDT')
            print(f"[Balance] BTC: {btc_balance:.6f}   USDT: {usdt_balance:.2f}")
            # ---------------------------------------------------------

            if not cont:
                break
            time.sleep(config.refresh_interval)
    except KeyboardInterrupt:
        logging.info("Manual interrupt received, stopping strategy.")
    finally:
        try:
            exchange.cancel_all_orders()
        except Exception as e:
            logging.error("Error during final order cancel: %s", e)
    logging.info("Live trading loop exited.")

def run_backtest():
    import numpy as np
    np.random.seed(42)
    price_series = 30000 + np.cumsum(np.random.normal(0, 50, 500))
    config = Config()
    exchange = SimulatedExchange(config, price_series)
    strategy = MarketMakerStrategy(exchange, config)
    logging.info("Starting backtest...")
    while True:
        cont = strategy.run_step()
        if not cont or exchange.current_index >= len(price_series):
            break
    final_inventory = exchange.inventory
    final_balance = exchange.balance
    starting_balance = exchange.starting_balance
    final_portfolio_value = final_balance + final_inventory * (exchange.current_price or 0)
    profit_loss = final_portfolio_value - starting_balance
    logging.info("Backtest finished. Final PnL: %.2f USDT (Inventory: %.4f BTC, Cash: %.2f USDT)",
                 profit_loss, final_inventory, final_balance)
    num_trades = len(exchange.trade_history)
    logging.info("Number of trades executed: %d", num_trades)

if __name__ == "__main__":
    # To run live trading, uncomment the next line:
    run_live()
    # To run backtest, uncomment the next line:
    # run_backtest()
