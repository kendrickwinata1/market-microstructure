class BacktestTradingStrategy:
    """
    Simulates backtesting a trading strategy over historical data.
    Handles trade execution, cash/BTC inventory, margin, leverage, and logs all trades.
    """

    def __init__(
        self,
        model,
        data,
        start_cash=10000,
        trading_lot=7500,
        stop_loss_threshold=0.004,
        leverage_factor=1,
        margin_call_threshold=0.5,
        annual_interest_rate=0.0,
    ):
        self.model = model
        self.data = data
        self.cash = start_cash
        self.starting_cash = start_cash
        self.margin_requirement = start_cash * margin_call_threshold
        self.trading_lot = trading_lot
        self.stop_loss_threshold = stop_loss_threshold
        self.leverage_factor = leverage_factor
        self.annual_interest_rate = annual_interest_rate

        self.trade_log = []
        self.buy_price = None
        self.btc_inventory = 0
        self.daily_return_factors = []
        self.interest_costs = []

    def execute_trades(self):
        """
        Execute trades using model predictions (expects model to have a .predict() method).
        """
        predicted_categories = self.model.predict()

        for (row_index, row), prediction in zip(self.data.iterrows(), predicted_categories):
            usd_btc_spot_rate = row["Close"]
            current_date = row["Timestamp"]

            print("spot rate:", usd_btc_spot_rate, "prediction:", prediction)

            is_stop_loss_triggered = self._check_stop_loss(usd_btc_spot_rate, current_date)

            # Attempt to buy BTC if predicted and enough cash is available
            if prediction == "Buy" and self.cash >= self.trading_lot:
                self._buy_btc(usd_btc_spot_rate, current_date)
            # Attempt to sell BTC if predicted and have inventory, and some profit
            elif (
                prediction == "Sell"
                and self.btc_inventory > 0
                and (
                    self.buy_price is None
                    or (self.buy_price is not None and usd_btc_spot_rate > self.buy_price * 1.003)
                )
            ):
                self._sell_btc(usd_btc_spot_rate, current_date)

    def execute_trades_perfect_future_knowledge(self):
        """
        Alternate execution with perfect foresight (uses "Label" as the trading signal).
        """
        for idx, row in self.data.iterrows():
            usd_btc_spot_rate = row["Open"]
            current_date = row["Date"]
            daily_change_pct = row["Daily_Change_Open_to_Close"]

            if self.btc_inventory > 0:
                self.daily_return_factors.append(1 + (daily_change_pct * self.leverage_factor))

            is_stop_loss_triggered = self._check_stop_loss(usd_btc_spot_rate, current_date)
            if is_stop_loss_triggered:
                continue

            # Logic: Buy on "Sell" label (assume buying USD/selling BTC), sell on "Buy" label
            if (
                row["Label"] == "Sell"
                and self.cash >= self.trading_lot
                and (
                    self.buy_price is None
                    or (
                        self.buy_price is not None and (
                            usd_btc_spot_rate < self.buy_price * 0.99
                            or usd_btc_spot_rate > self.buy_price * 1.01
                        )
                    )
                )
            ):
                self._buy_btc(usd_btc_spot_rate, current_date)
            elif row["Label"] == "Buy" and self.btc_inventory > 0:
                self._sell_btc(usd_btc_spot_rate, current_date)

            if self._check_margin_call(usd_btc_spot_rate):
                print("MARGIN CALL!!! this should not happen!")
                self._sell_btc(usd_btc_spot_rate, current_date)

    def _buy_btc(self, rate, date):
        """
        Executes a BTC buy, updates inventory and cash, and logs the trade.
        """
        print(f"Buying BTC at rate: {rate} on {date}")
        btc_bought = self.trading_lot * self.leverage_factor / rate
        print(f"Amount bought: {btc_bought}")
        self.btc_inventory += btc_bought
        self.cash -= self.trading_lot
        self.buy_price = rate
        self.trade_log.append(f"Buy {btc_bought} btc at {rate} on {date}")

    def _sell_btc(self, rate, date, forced=False):
        """
        Sells all BTC inventory, updates cash to mark-to-market, and logs the trade.
        """
        if self.btc_inventory <= 0:
            return
        print(f"Selling BTC at rate: {rate} on {date}")
        self.cash = self._compute_mtm(rate)
        reason = "Model predicted sell" if not forced else "Margin call / stop-loss triggered"
        self.trade_log.append(
            f"Sell {self.btc_inventory} btc at {rate} on {date} ({reason})"
        )
        self._apply_interest_charge(rate)
        self.btc_inventory = 0
        self.daily_return_factors = []

    def _compute_mtm(self, usd_btc_spot_rate):
        """
        Calculate mark-to-market portfolio value.
        """
        if self.btc_inventory <= 0:
            return self.cash

        current_value = self.btc_inventory * usd_btc_spot_rate
        invested_amount = self.btc_inventory * self.buy_price
        pnl = current_value - invested_amount
        principal = self.trading_lot
        mtm = self.cash + principal + pnl
        print(f"Mark-to-market: {mtm}")
        return mtm

    def _check_stop_loss(self, usd_btc_spot_rate, date):
        """
        Triggers stop loss if price drops below threshold since last buy.
        """
        if self.btc_inventory > 0 and self.buy_price:
            change_pct = abs(usd_btc_spot_rate - self.buy_price) / self.buy_price
            if change_pct * self.leverage_factor > self.stop_loss_threshold:
                self._sell_btc(usd_btc_spot_rate, date, forced=True)
                return True
        return False

    def _check_margin_call(self, usd_btc_spot_rate):
        """
        Triggers margin call if portfolio value drops below maintenance margin.
        """
        if self.btc_inventory > 0:
            if self._compute_mtm(usd_btc_spot_rate) < self.margin_requirement:
                return True
        return False

    def _apply_interest_charge(self, rate):
        """
        Applies daily interest charges on borrowed funds, if any.
        """
        days_held = len(self.daily_return_factors)
        daily_rate = (1 + self.annual_interest_rate) ** (1 / 365) - 1
        borrowed = self.btc_inventory - (self.btc_inventory / self.leverage_factor)
        interest = (borrowed / rate) * daily_rate * days_held
        self.interest_costs.append(interest)

    def evaluate_performance(self):
        """
        Computes portfolio stats at end of backtest.
        """
        final_rate = self.data.iloc[-1]["Close"]
        final_value = self._compute_mtm(final_rate)
        print(f"Final portfolio value: {final_value}")
        print(f"Final BTC inventory: {self.btc_inventory}")
        pnl_per_trade = ((final_value - self.starting_cash) / len(self.trade_log)) if self.trade_log else 0
        print(f"Interest costs: {self.interest_costs}")
        return {
            "Final Portfolio Value": final_value,
            "Number of Trades": len(self.trade_log),
            "Profit/Loss per Trade": pnl_per_trade,
            "Trade Log": self.trade_log,
            "Interest Costs": self.interest_costs,
            "Transaction Costs": 0,  # Assume zero for now
        }
