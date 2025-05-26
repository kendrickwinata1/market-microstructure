import pandas as pd
import time
from urllib.parse import urlencode
import hmac
import hashlib
import requests

class BookKeeper:
    """
    BookKeeper tracks all executed trades, account balances, and positions for the trading system.
    It can provide historical data for P&L analysis, risk metrics, and visualization.
    """

    def __init__(self, symbol, api_key=None, api_secret=None):
        # Endpoint for Binance Futures Testnet (change as needed for production)
        self.BASE_URL = "https://testnet.binancefuture.com"
        self._api_key = api_key
        self._api_secret = api_secret
        self.symbol = symbol

        # DataFrames for time series tracking
        self.market_prices = pd.DataFrame(columns=["Date", "Symbol", "Price"])
        self.historical_data = pd.DataFrame(columns=[
            "Timestamp",
            "WalletBalance",
            "AvailableBalance",
            "RealizedProfit",
            "UnrealizedProfit",
        ])
        self.historical_positions = pd.DataFrame(columns=[
            "Timestamp",
            "Symbol",
            "entryPrice",
            "PositionAmt",
        ])

        # Get initial wallet balance from the exchange on initialization
        timestamp = int(time.time() * 1000)
        params = {"timestamp": timestamp}
        query_string = urlencode(params)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        url = f"{self.BASE_URL}/fapi/v2/account?{query_string}&signature={signature}"

        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json;charset=utf-8",
            "X-MBX-APIKEY": self._api_key,
        })
        response = session.get(url=url, params={})
        response_map = response.json()
        self.initial_cash = response_map["availableBalance"]

    # --- Properties to quickly access latest PnL and balances ---

    @property
    def get_unrealized_pnl(self):
        """Return latest unrealized P&L value."""
        return self.historical_data["UnrealizedProfit"].iloc[-1]

    @property
    def get_realized_pnl(self):
        """Return latest realized P&L value."""
        return self.historical_data["RealizedProfit"].iloc[-1]

    @property
    def get_wallet_balance(self):
        """Return latest wallet balance value."""
        return self.historical_data["AvailableBalance"].iloc[-1]

    # --- Main method to update state on every tick or trade event ---

    def update_bookkeeper(self, date, middle_price, timestamp):
        """
        On every tick or relevant event, update:
        1. Latest price
        2. Account balances (wallet, available, realized/unrealized PnL)
        3. Position size and entry price
        """
        # API call to get current wallet and position data
        params = {"timestamp": timestamp}
        query_string = urlencode(params)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        url = f"{self.BASE_URL}/fapi/v2/account?{query_string}&signature={signature}"

        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json;charset=utf-8",
            "X-MBX-APIKEY": self._api_key,
        })
        response = session.get(url=url, params={})
        response_map = response.json()

        # If too much data, trim (keep last 86400 rows max)
        if self.historical_data.shape[0] == 86400:
            self.market_prices = self.market_prices.iloc[1:, :]
            self.historical_data = self.historical_data.iloc[1:, :]
            self.historical_positions = self.historical_positions.iloc[1:, :]

        # Update market price DataFrame
        temp = pd.Series(
            data=[date, self.symbol, middle_price],
            index=["Date", "Symbol", "Price"]
        )
        self.market_prices = pd.concat(
            [self.market_prices, temp.to_frame().T], ignore_index=True
        )

        # Update historical_data DataFrame (account, P&L)
        temp = pd.Series(
            data=[
                timestamp,
                float(response_map["totalWalletBalance"]),
                float(response_map["availableBalance"]),
                float(response_map["totalWalletBalance"]) - float(self.initial_cash),
                float(response_map["totalUnrealizedProfit"]),
            ],
            index=[
                "Timestamp",
                "WalletBalance",
                "AvailableBalance",
                "RealizedProfit",
                "UnrealizedProfit",
            ],
        )
        self.historical_data = pd.concat(
            [self.historical_data, temp.to_frame().T], ignore_index=True
        )

        # Update position DataFrame for this symbol
        temp = pd.DataFrame(response_map["positions"])
        temp = temp[temp["symbol"] == self.symbol]
        temp = pd.Series(
            data=[
                timestamp,
                self.symbol,
                float(temp["entryPrice"].iloc[0]),
                float(temp["positionAmt"].iloc[0]),
            ],
            index=["Timestamp", "Symbol", "entryPrice", "PositionAmt"],
        )
        self.historical_positions = pd.concat(
            [self.historical_positions, temp.to_frame().T], ignore_index=True
        )

    # --- Performance and risk metrics ---

    def calculate_max_drawdown(self):
        """Return the maximum observed drawdown based on wallet balance history."""
        roll_max = self.historical_data["WalletBalance"].cummax()
        max_daily_drawdown = (
            self.historical_data["WalletBalance"] / roll_max - 1.0
        ).cummin()
        return max_daily_drawdown.iloc[-1]

    def calculate_sharpe_ratio(self, risk_free_rate=0):
        """Return the Sharpe ratio based on wallet balance changes."""
        returns = self.historical_data["WalletBalance"].pct_change().dropna()
        if returns.std() == 0:
            return 0  # Avoid division by zero
        sharpe = (returns.mean() - risk_free_rate) / returns.std()
        return sharpe

    def calculate_vol(self):
        """Return standard deviation (volatility) of market price history."""
        return self.market_prices["Price"].dropna().std()

    # --- Data export helpers ---

    def return_historical_data(self):
        """Return the full DataFrame of account data (balances, P&L, etc.)."""
        return self.historical_data

    def return_historical_market_prices(self):
        """Return the full DataFrame of observed market prices."""
        return self.market_prices

    def return_historical_positions(self):
        """Return the full DataFrame of observed historical positions."""
        return self.historical_positions
