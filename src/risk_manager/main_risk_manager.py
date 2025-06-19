import logging

class RiskManager:
    """
    The RiskManager class monitors and controls risk exposure by:
    - Tracking profit & loss, mark-to-market, and risk metrics
    - Checking for stop loss and drawdown triggers
    - Providing balance/position checks to the trading engine
    """

    def __init__(self, book_keeper):
        """
        Initialize the risk manager with a BookKeeper instance.

        Args:
            book_keeper: The BookKeeper object for accessing trade/portfolio history.
        """
        self.book_keeper = book_keeper
        self.risk_metrics = {}
        self.greeks = {}  # Placeholder: Track option greeks if required in the future

    def check_available_balance(self, trade):
        """
        Check if there's enough available balance (after minimum cash ratio)
        to execute a new buy trade.

        Args:
            trade: Total dollar amount intended for buying.
        Returns:
            bool: True if trade allowed, False otherwise.
        """
        print(f"buy trade check amt{trade}")
        historical_data_df = self.book_keeper.return_historical_data()
        current_available_balance = float(historical_data_df["AvailableBalance"].iloc[-1])
        print(f"[risk mgr] AvailableBalance :{current_available_balance}")

        current_portfolio_balance = float(historical_data_df["WalletBalance"].iloc[-1])
        print(f"[risk mgr] WalletBalance :{current_available_balance}")

        minimum_cash_ratio = 0.25
        post_trade_cash_ratio = (
            current_available_balance - trade
        ) / current_portfolio_balance
        post_trade_cash_ratio = round(post_trade_cash_ratio, 2)
        print(f"post trade ratio {post_trade_cash_ratio}")

        # Only allow trade if post-trade cash ratio remains above minimum
        return post_trade_cash_ratio >= minimum_cash_ratio

    def get_available_tradable_balance(self):
        """
        Return the available balance for trading after enforcing a minimum cash ratio.

        Returns:
            float: Amount of cash allowed to be used for new trades.
        """
        historical_data_df = self.book_keeper.return_historical_data()
        current_available_balance = float(historical_data_df["AvailableBalance"].iloc[-1])
        print(f"[risk mgr] AvailableBalance :{current_available_balance}")
        minimum_cash_ratio = 0.25
        available_trade_balance = (1 - minimum_cash_ratio) * current_available_balance
        return available_trade_balance

    def get_last_buy_price(self):
        """
        Find the most recent valid buy price by inspecting position increases.

        Returns:
            float or None: Last buy price if valid; otherwise None.
        """
        historical_positions_df = self.book_keeper.return_historical_positions()

        if historical_positions_df.empty or "PositionAmt" not in historical_positions_df.columns:
            logging.error("[RiskManager] Historical positions dataframe invalid or empty.")
            return None

        buy_transactions = historical_positions_df[
            historical_positions_df["PositionAmt"].diff() > 0
        ]

        if not buy_transactions.empty:
            last_buy_price = buy_transactions["entryPrice"].iloc[-1]
            if last_buy_price > 0:
                return last_buy_price

        logging.warning("[RiskManager] No valid buy transactions found.")
        return None


    def get_current_btc_inventory(self):
        """
        Return the most current BTC position amount.
        Returns:
            float: Current BTC position (can be 0 if flat).
        """
        historical_positions_df = self.book_keeper.return_historical_positions()
        return historical_positions_df["PositionAmt"].iloc[-1]

    def check_buy_position(self):
        """
        Only allow new buys if no existing position (flat or after previous sell).
        Returns:
            bool: True if buying is allowed, else False.
        """
        return float(self.get_current_btc_inventory()) == 0

    def check_sell_position(self):
        """
        Only allow selling if a long position exists.
        Returns:
            bool: True if selling is allowed, else False.
        """
        return float(self.get_current_btc_inventory()) > 0

    def trigger_stop_loss(self):
        """
        Trigger stop loss if price falls below a set threshold from last buy price.
        Returns:
            bool: True if stop loss should be triggered, else False.
        """
        last_buy_price = self.get_last_buy_price()
        if last_buy_price is None or last_buy_price <= 0:
            print("Cannot trigger stop loss due to invalid last buy price")
            return False

        market_price_df = self.book_keeper.return_historical_market_prices()
        latest_market_price = market_price_df["Price"].iloc[-1]
        current_btc_inventory = self.get_current_btc_inventory()

        stoploss_threshold = 0.01  # E.g. 1% stop loss
        stoploss_limit_value = (1 - stoploss_threshold) * last_buy_price
        print(
            f"STOPLOSS CHECK: stoploss_limit_value{stoploss_limit_value}, "
            f"latest_market_price{latest_market_price},  last_buy_price{last_buy_price}"
        )
        stoploss_limit_value = float(stoploss_limit_value)
        latest_market_price = float(latest_market_price)

        if current_btc_inventory > 0:
            return latest_market_price <= stoploss_limit_value
        else:
            return False

    def trigger_trading_halt(self):
        """
        Trigger liquidation if the drawdown exceeds the threshold.

        Returns:
            bool: True if trading halt/liquidation should be triggered.
        """
        daily_maxdrawdown = self.book_keeper.calculate_max_drawdown()
        daily_mdd_threshold = -0.05  # e.g. 5% max drawdown
        current_btc_inventory = self.get_current_btc_inventory()
        if current_btc_inventory > 0:
            return daily_maxdrawdown <= daily_mdd_threshold
        else:
            return False

    def check_short_position(self, ordersize):
        """
        Check if it's allowed to place a sell order (short).
        Only allow if not exceeding current inventory.

        Args:
            ordersize: Proposed order size (BTC)
        Returns:
            bool: True if short is allowed, else False.
        """
        current_btc_inventory = self.get_current_btc_inventory()
        print(f"current_btc_inventory {current_btc_inventory} vs ordersize {ordersize}")
        if current_btc_inventory == 0 and ordersize == 0:
            print("All is Zero, nothing to do")
        elif ordersize <= current_btc_inventory:
            return True
        else:
            print("No short position allowed")
            return False

    def check_buy_order_value(self, buyprice):
        """
        Check if a buy order price is within allowed limits.

        Args:
            buyprice: Proposed buy price
        Returns:
            bool: True if order price allowed, else False.
        """
        market_price_df = self.book_keeper.return_historical_market_prices()
        latest_market_price = float(market_price_df["Price"].iloc[-1])
        upper_buy_price_ratio = 1.1   # Don't buy if >10% above market
        lower_buy_price_ratio = 0.6   # Don't buy if <60% of market (likely error)
        if (
            lower_buy_price_ratio * latest_market_price <= buyprice
            and buyprice <= latest_market_price * upper_buy_price_ratio
        ):
            return True
        else:
            print("Check buy order value")
            return False

    def check_sell_order_value(self, sellprice):
        """
        Check if a sell order price is within allowed limits and above minimum profit.

        Args:
            sellprice: Proposed sell price
        Returns:
            bool: True if order price allowed, else False.
        """
        last_buy_price = self.get_last_buy_price() or 0
        market_price_df = self.book_keeper.return_historical_market_prices()
        latest_market_price = float(market_price_df["Price"].iloc[-1])
        min_sell_threshold = 1.01  # Only allow selling above last buy by at least 1%

        lower_sell_price_ratio = 0.9  # Don't sell too low
        upper_sell_price_ratio = 1.4  # Don't sell too high

        if (
            latest_market_price * lower_sell_price_ratio <= sellprice
            and sellprice <= latest_market_price * upper_sell_price_ratio
            and sellprice >= last_buy_price * min_sell_threshold
        ):
            return True
        else:
            print("Check sell order value")
            return False

    @staticmethod
    def check_symbol(tradesymbol):
        """
        Ensure we're only trading the allowed symbol to avoid accidental trades.

        Args:
            tradesymbol: Symbol for the trade (e.g. 'BTCUSDT')
        Returns:
            bool: True if symbol is allowed.
        """
        ticker_control = "BTCUSDT"
        if tradesymbol == ticker_control:
            return True
        else:
            print("Check trade symbol")
            return False
