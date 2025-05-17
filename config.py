# config.py
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Config:
    # Binance Testnet API Keys (replace with your own)
    api_key: str = os.getenv("BINANCE_API_KEY")
    api_secret: str = os.getenv("BINANCE_API_SECRET")

    # Trading Parameters
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"

    spread_pct: float = 0.001        # 0.1% spread
    order_size: float = 0.01         # 0.01 BTC per side
    inventory_target: float = 0.0    # Neutral
    inventory_skew_factor: float = 1.0
    refresh_interval: float = 5.0    # seconds
    position_limit: float = 0.05     # max BTC long/short
    loss_threshold: float = 100.0    # USDT
    initial_balance: float = 10000.0 # for backtest
    initial_inventory: float = 0.0   # for backtest
