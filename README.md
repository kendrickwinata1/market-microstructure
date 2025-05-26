Crypto Trading Bot – Modular Algorithmic System for Binance Futures
This project is a modular, event-driven algorithmic trading bot for the Binance Futures Testnet. It features live market data streaming, automated ML-based strategy execution, robust risk management, real-time bookkeeping, and order management.

Project Features
Live market data streaming (websocket)

Automated trade execution (Buy/Sell/Hold signals from ML or rule-based models)

Bookkeeping: Real-time PnL, balance, and audit tracking

Risk management: Stop-loss, max drawdown, and order sanity checks

Order management: Limit/market orders, order aging/cancellation

Backtesting and model training (offline utilities)

Rich feature engineering and visualization tools

Project Structure
.
├── app.py                           # Main trading loop and orchestrator
├── gateway/
│   ├── data_stream.py               # Live Binance websocket data stream
│   ├── market_data_stream.py        # Simpler websocket market data stream
│   └── main_gateway.py              # Trade execution API logic
├── book_keeper/
│   └── main_book_keeper.py          # BookKeeper: PnL, balance, positions
├── risk_manager/
│   └── main_risk_manager.py         # RiskManager: checks stop-loss, limits
├── trading_engine/
│   └── main_trading_strategy.py     # TradingStrategy: model & signal generation
├── rest_connect/
│   └── rest_factory.py              # REST API connection factory for Binance
├── base_model.py                    # Abstract base ML model class
├── logreg_model.py                  # Logistic Regression ML model
├── backtest_trading_strategy.py     # Backtest engine for historical simulation
├── model_old.py                     # Feature engineering, TDA labeling
├── visualize.py                     # Visualization utility
├── live_plotter.py                  # Live matplotlib price plotting
├── correlation.py                   # Research: feature/signal correlation
├── fetch_historical.py              # Fetches historical trades via CCXT
├── historical.py                    # Fetches OHLCV data via CCXT
├── pnl.py                           # Simple script for total PnL from signals
└── .env                             # (Not included) Binance API credentials

Main Application Workflow (app.py)
1. Startup
Loads API credentials from .env

Initializes:

REST gateway for Binance

TradeExecutor (for order sending/cancelling)

BookKeeper (PnL/position tracking)

RiskManager (for stop-loss, limits, checks)

TradingStrategy (signal generation)

DataStream (live websocket from Binance)

ExecManager (main orchestrator: connects all components)

2. Event-Driven Trading (Live Loop)
DataStream streams market data. Each new tick triggers ExecManager.exec_strat.

ExecManager handles:

Updates BookKeeper with new prices and balances

Cancels any stale open orders

Checks RiskManager for stop-loss, drawdown, or trading halt (liquidates if needed)

Runs TradingStrategy to get next action (Buy/Sell/Hold + price)

If trade is signaled, checks all risk rules before sending order via TradeExecutor

All order events are recorded and output to logs/CSV

Order/portfolio changes are continuously saved for audit/tracking.

3. Heartbeat
Prints "application running" every 10 seconds to show liveness.

⚡ Offline Utilities (Development/Analysis)
Model Training & Feature Engineering

base_model.py, logreg_model.py, model_old.py

Backtesting

backtest_trading_strategy.py: Simulate strategy performance on historical data

Visualization

visualize.py: Plot price and model features

live_plotter.py: Live charting (Tkinter + Matplotlib)

PnL Analysis

pnl.py: Computes total profit/loss from labeled signal files

Historical Data Download

fetch_historical.py, historical.py: Download trades or OHLCV data for research

🔌 How the Modules Communicate
DataStream receives live ticks, calls ExecManager.exec_strat every update

ExecManager:

Updates BookKeeper

Calls RiskManager for safety checks

Runs TradingStrategy for signal

Places/cancels orders using TradeExecutor/RestGateway

RiskManager: Decides if trades are allowed, or if liquidation is needed

BookKeeper: Tracks balances, PnL, positions, and saves to CSV

🛠️ Setup & Running
Install dependencies
pip install -r requirements.txt

Set up your .env file with your Binance API credentials:
API_KEY=your_binance_testnet_key
API_SECRET=your_binance_testnet_secret

Run the bot:
python app.py

