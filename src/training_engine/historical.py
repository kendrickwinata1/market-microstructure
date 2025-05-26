import time
import ccxt
import pandas as pd

# Initialize Binance exchange with proper rate limit settings
exchange = ccxt.binance({
    "rateLimit": 1200,
    "enableRateLimit": True,
})

def fetch_binance_ohlcv(symbol, timeframe, since, limit=500):
    """
    Fetch historical OHLCV data from Binance in batches.
    Args:
        symbol (str): The trading symbol, e.g., 'BTC/USDT'
        timeframe (str): Granularity, e.g., '1m', '5m', '1h'
        since (int): Timestamp in ms to start from
        limit (int): Number of candles per API call (max 1000 for Binance, default 500)
    Returns:
        list: List of OHLCV candles
    """
    all_data = []
    while True:
        data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not data:
            break
        all_data.extend(data)
        since = data[-1][0] + 1  # Start from next millisecond after last candle
        time.sleep(exchange.rateLimit / 1000)
        if len(data) < limit:
            break  # No more data left
    return all_data

if __name__ == "__main__":
    # --- Parameters ---
    symbol = "BTC/USDT"
    timeframe = "1m"  # 1-minute bars
    since = exchange.parse8601("2024-04-14T00:00:00Z")  # Start datetime

    # --- Fetch data ---
    ohlcv_data = fetch_binance_ohlcv(symbol, timeframe, since)

    # --- Convert to DataFrame ---
    columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    df = pd.DataFrame(ohlcv_data, columns=columns)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")

    # --- Save to CSV ---
    out_csv = "inputs/binance_btcusdt_1min_ccxt.csv"
    df.to_csv(out_csv, index=False)
    print(f"Data saved to {out_csv}")
