import ccxt
import pandas as pd
from datetime import datetime, timedelta


def fetch_trades_to_ohlc(symbol, hours=2, resample_interval='1s', csv_out="ohlc_2hrs.csv"):
    """
    Fetch recent trades for a symbol from Binance USDT Futures and aggregate to OHLCV DataFrame.

    Args:
        symbol (str): Trading pair symbol, e.g., 'BTC/USDT'
        hours (int): How many past hours of data to fetch
        resample_interval (str): Pandas resample frequency, e.g., '1s' for 1-second bars
        csv_out (str): Path to output CSV file
    """
    # Initialize exchange (anonymous)
    exchange = ccxt.binanceusdm()

    # Define time range (since = 2 hours ago)
    since_dt = datetime.utcnow() - timedelta(hours=hours)
    since = exchange.parse8601(since_dt.isoformat())

    all_trades = []
    print(f"Fetching trades for {symbol} since {since_dt.isoformat()} ...")
    # Loop to fetch all trades until now
    while since < exchange.milliseconds():
        trades = exchange.fetch_trades(symbol, since=since, limit=500)
        if not trades:
            break
        all_trades.extend(trades)
        since = trades[-1]['timestamp'] + 1  # Continue from the last trade

    if not all_trades:
        print("No trades fetched.")
        return

    # Convert to DataFrame
    trades_df = pd.DataFrame(all_trades)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='ms')

    # Aggregate trades into OHLCV bars
    ohlc = trades_df.resample(resample_interval, on='timestamp').agg({
        'price': ['first', 'max', 'min', 'last'],
        'amount': 'sum'
    }).dropna()
    ohlc.columns = ['open', 'high', 'low', 'close', 'volume']

    print(ohlc)
    ohlc.to_csv(csv_out)
    print(f"Saved to {csv_out}")

    # # Optional: Plot (uncomment if needed)
    # import matplotlib.pyplot as plt
    # fig, ax1 = plt.subplots(figsize=(12, 6))
    # ohlc[['open', 'high', 'low', 'close']].plot(ax=ax1)
    # ax1.set_ylabel('Price')
    # ax1.set_title(f'OHLC Data for {symbol}')
    # ax1.legend(['Open', 'High', 'Low', 'Close'])
    # ax2 = ax1.twinx()
    # ohlc['volume'].plot(kind='bar', ax=ax2, alpha=0.3, color='gray', width=0.03, position=0)
    # ax2.set_ylabel('Volume')
    # plt.show()


if __name__ == "__main__":
    fetch_trades_to_ohlc(symbol='BTC/USDT', hours=2, resample_interval='1s', csv_out="ohlc_2hrs.csv")
