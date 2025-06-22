# FILE: main_risk_manager.py
import logging


class RiskManager:
    """
    MODIFIED: The RiskManager class now controls risk for both LONG and SHORT positions.
    """

    def __init__(self, book_keeper):
        self.book_keeper = book_keeper
        self.risk_metrics = {}
        self.greeks = {}

    # --- Balance and Inventory Checks (No Changes Needed) ---
    def check_available_balance(self, trade):
        historical_data_df = self.book_keeper.return_historical_data()
        if historical_data_df.empty: return False
        current_available_balance = float(historical_data_df["AvailableBalance"].iloc[-1])
        current_portfolio_balance = float(historical_data_df["WalletBalance"].iloc[-1])
        minimum_cash_ratio = 0.25
        if current_portfolio_balance == 0: return False
        post_trade_cash_ratio = (current_available_balance - trade) / current_portfolio_balance
        post_trade_cash_ratio = round(post_trade_cash_ratio, 2)
        return post_trade_cash_ratio >= minimum_cash_ratio

    def get_available_tradable_balance(self):
        historical_data_df = self.book_keeper.return_historical_data()
        if historical_data_df.empty: return 0.0
        current_available_balance = float(historical_data_df["AvailableBalance"].iloc[-1])
        minimum_cash_ratio = 0.25
        return (1 - minimum_cash_ratio) * current_available_balance

    def get_current_btc_inventory(self):
        historical_positions_df = self.book_keeper.return_historical_positions()
        if historical_positions_df.empty: return 0.0
        return float(historical_positions_df["PositionAmt"].iloc[-1])

    # --- Position Entry Price Checks (MODIFIED and NEW) ---
    def get_last_buy_price(self):
        """(MODIFIED) Finds the entry price of the last entered long position."""
        historical_positions_df = self.book_keeper.return_historical_positions()
        if historical_positions_df.empty: return None
        buy_transactions = historical_positions_df[
            (historical_positions_df["PositionAmt"] > 0) & (historical_positions_df["PositionAmt"].shift(1) <= 0)
            ]
        if not buy_transactions.empty:
            return buy_transactions["entryPrice"].iloc[-1]
        return None

    def get_last_short_entry_price(self):
        """(NEW) Finds the entry price of the last entered short position."""
        historical_positions_df = self.book_keeper.return_historical_positions()
        if historical_positions_df.empty: return None
        short_transactions = historical_positions_df[
            (historical_positions_df["PositionAmt"] < 0) & (historical_positions_df["PositionAmt"].shift(1) >= 0)
            ]
        if not short_transactions.empty:
            return short_transactions["entryPrice"].iloc[-1]
        return None

    # --- Position Directional Checks (MODIFIED) ---
    def check_buy_position(self):
        """(MODIFIED) Allow buying only if flat or currently short (to cover)."""
        return self.get_current_btc_inventory() <= 0

    def check_sell_position(self):
        """(MODIFIED) Allow selling only if flat or currently long (to exit)."""
        return self.get_current_btc_inventory() >= 0

    # --- Risk Trigger Checks (MODIFIED and NEW) ---
    def trigger_stop_loss(self):
        """(MODIFIED) Triggers stop loss for EITHER a long or a short position."""
        current_btc_inventory = self.get_current_btc_inventory()
        if current_btc_inventory == 0: return False

        market_price_df = self.book_keeper.return_historical_market_prices()
        if market_price_df.empty: return False
        latest_market_price = float(market_price_df["Price"].iloc[-1])
        stoploss_threshold = 0.01

        if current_btc_inventory > 0:
            last_buy_price = self.get_last_buy_price()
            if not last_buy_price: return False
            stoploss_limit = (1 - stoploss_threshold) * last_buy_price
            logging.info(f"STOPLOSS (LONG) CHECK: Price < {stoploss_limit} (Entry: {last_buy_price})")
            return latest_market_price <= stoploss_limit

        elif current_btc_inventory < 0:
            last_short_price = self.get_last_short_entry_price()
            if not last_short_price: return False
            stoploss_limit = (1 + stoploss_threshold) * last_short_price
            logging.info(f"STOPLOSS (SHORT) CHECK: Price > {stoploss_limit} (Entry: {last_short_price})")
            return latest_market_price >= stoploss_limit
        return False

    def trigger_trading_halt(self):
        daily_maxdrawdown = self.book_keeper.calculate_max_drawdown()
        daily_mdd_threshold = -0.05
        return self.get_current_btc_inventory() != 0 and daily_maxdrawdown <= daily_mdd_threshold

    def check_short_position(self, ordersize):
        """This function checks if we can sell our existing long inventory."""
        return ordersize <= self.get_current_btc_inventory()

    # --- Order Value Sanity Checks (MODIFIED and NEW) ---
    def check_buy_order_value(self, buyprice):
        market_price_df = self.book_keeper.return_historical_market_prices()
        if market_price_df.empty: return False
        latest_market_price = float(market_price_df["Price"].iloc[-1])
        if not (0.9 * latest_market_price <= buyprice <= 1.1 * latest_market_price):
            logging.warning(
                f"Check order value FAILED: Price {buyprice} is >10% away from market {latest_market_price}")
            return False
        return True

    def check_sell_order_value(self, sellprice):
        last_buy_price = self.get_last_buy_price() or 0
        min_sell_threshold = 0.999
        if last_buy_price > 0 and sellprice >= last_buy_price * min_sell_threshold:
            return True
        logging.warning(f"Check sell value FAILED: Sell price {sellprice} not profitable vs buy price {last_buy_price}")
        return False

    def check_buy_to_cover_value(self, buyprice):
        """(NEW) Check if a BUY order to CLOSE a short position is profitable."""
        last_short_price = self.get_last_short_entry_price() or 0
        if last_short_price == 0: return False
        min_profit_threshold = 1.001
        if buyprice <= last_short_price * min_profit_threshold:
            return True
        logging.warning(
            f"Check buy to cover FAILED: Buy price {buyprice} not profitable vs short price {last_short_price}")
        return False