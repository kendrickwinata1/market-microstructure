import tkinter as tk  # Imported for potential GUI extension; not used directly here.
from queue import Queue  # Used for thread-safe data exchange.
from threading import Thread  # Enables concurrent data fetching.

from book_keeper.main_book_keeper import BookKeeper   # Trade and position logging (not used directly here).
from gateway.market_data_stream import MarketDataStream  # Handles market data acquisition.
from gateway.main_gateway import TradeExecutor  # For executing trades (not used directly here).
from risk_manager.main_risk_manager import RiskManager  # Risk controls (not used directly here).
from trading_engine.main_trading_strategy import TradingStrategy  # Main trading logic.
import schedule  # For periodic task scheduling.
import time  # Timing and sleep control.

if __name__ == "__main__":
    # Initialize a thread-safe queue for passing market data between threads.
    queue = Queue()

    # Instantiate the trading strategy and market data stream components.
    # Both components use the shared queue for data flow.
    strategy = TradingStrategy(queue)
    data_stream = MarketDataStream(queue)

    # Start background thread for continuous data fetching without blocking the main loop.
    data_thread = Thread(target=data_stream.fetch_data, daemon=True)
    data_thread.start()

    time_elapsed = 0  # Tracks total runtime in seconds.

    def job():
        """
        Scheduled job executed every second:
        - Aggregates the data collected so far for feature computation.
        - After sufficient data/time, triggers model analysis and prints the prediction.
        """
        strategy.aggregate_data()
        if time_elapsed > 40:
            output = strategy.analyze_data()
            print("model output:", output)

    # Register the scheduled job to run every second.
    schedule.every().second.do(job)

    # Main event loop:
    #  - Continuously collects new market data from the queue.
    #  - Runs any scheduled tasks (such as aggregation and model inference).
    #  - Maintains a regular timing cadence.
    while True:
        # Poll the queue and integrate new data into the strategy’s dataset.
        strategy.collect_new_data()

        # Execute scheduled jobs (aggregation/model prediction).
        schedule.run_pending()

        # Control loop pace; avoids excessive CPU usage.
        time.sleep(0.5)

        # Increment and display elapsed time for monitoring/logging.
        time_elapsed += 0.5
        print("time_elapsed:", time_elapsed)
