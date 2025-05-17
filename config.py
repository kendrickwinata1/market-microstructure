from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load env variables for API keys and sensitive info
load_dotenv()

@dataclass
class Config:
    # General
    mode: str = os.getenv("MODE", "live")  # "live" or "backtest"
    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    leverage: int = int(os.getenv("LEVERAGE", 1))
    # API
    api_key: str = os.getenv("BINANCE_API_KEY")
    api_secret: str = os.getenv("BINANCE_API_SECRET")
    # Trading params
    spread_pct: float = float(os.getenv("SPREAD_PCT", 0.10))     # 0.10% spread
    order_size: float = float(os.getenv("ORDER_SIZE", 0.001))    # Futures: minimum for BTCUSDT is 0.001
    position_limit: float = float(os.getenv("POSITION_LIMIT", 0.01))
    inventory_target: float = float(os.getenv("INVENTORY_TARGET", 0.0))
    inventory_skew_factor: float = float(os.getenv("INVENTORY_SKEW", 2.0))
    refresh_interval: float = float(os.getenv("REFRESH_INTERVAL", 5.0))  # in seconds
    max_drawdown: float = float(os.getenv("MAX_DRAWDOWN", 20.0))         # Stop if PnL < -$20
    # Backtest
    use_random_walk: bool = os.getenv("USE_RANDOM_WALK", "True") == "True"
    price_series_file: str = os.getenv("PRICE_SERIES_FILE", "")
    # Visualization
    enable_plotting: bool = os.getenv("ENABLE_PLOTTING", "True") == "True"
    # Logging
    log_file: str = os.getenv("LOG_FILE", "market_maker.log")
