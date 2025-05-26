# Market Microstructure Trading Engine

## Overview

This project is a modular Python framework for developing, backtesting, and running live trading strategies for cryptocurrency (specifically BTC/USDT) on Binance Futures (testnet).

The architecture integrates live data streaming, REST API execution, order management, ML-based signal generation, risk management, bookkeeping, backtesting, and visualization tools.

---

## Folder Structure

```

market-microstructure/

│   app.py

│   README.md

│

├── gateway/

│   ├── data_stream.py

│   ├── market_data_stream.py

│   └── main_gateway.py

├── rest_connect/

│   └── rest_factory.py

├── book_keeper/

│   └── main_book_keeper.py

├── risk_manager/

│   └── main_risk_manager.py

├── trading_engine/

│   └── main_trading_strategy.py

│

├── notebooks/

├── inputs/

├── outputs/

│

├── model/

│   ├── base_model.py

│   ├── logreg_model.py

│   └── model_old.py

├── analysis/

│   ├── correlation.py

│   ├── pnl.py

│   └── visualize.py

│

├── fetch_historical.py

├── historical.py

├── backtest_trading_strategy.py

├── review_engine.py

├── live_plotter.py

└── README.md

```

---

## Main Modules

- **app.py**: Main application entry point, orchestrates system objects, data streaming, trading loop, and heartbeat.

- **gateway/**: Handles real-time market data (websocket) and trade execution (REST API).

- **book_keeper/**: Stores and updates account balance, positions, and historical PnL.

- **risk_manager/**: Implements risk limits, order approval, stop-loss, and trading halt logic.

- **trading_engine/**: Core trading logic (signal generation, data aggregation, order signals).

- **rest_connect/**: REST API connector and abstraction for exchange endpoints.

- **model/**: Machine learning models for market prediction (e.g., logistic regression, signal labelling, feature engineering).

- **analysis/**: Utilities for correlation, PnL calculation, visualization, and exploratory analysis.

- **fetch_historical.py / historical.py**: Download and preprocess historical data for model training or backtesting.

- **backtest_trading_strategy.py**: Backtesting framework for evaluating trading strategy performance on historical data.

- **review_engine.py**: Manages retraining, backtesting, and updating of the ML model.

- **live_plotter.py**: (Optional) Real-time plotting of price data, peaks, troughs, and signal events.

---

## How the Components Communicate (app.py-centric)

1. **app.py** loads API keys and sets up main components:

   - `RestFactory` for REST API

   - `TradeExecutor` for sending orders

   - `BookKeeper` for state/history

   - `RiskManager` for all risk controls

   - `ExecManager` which brings them all together

   - `DataStream` for websocket tick data

2. **Data Flow:**

   - DataStream receives live price ticks (via websocket) and calls `exec_manager.exec_strat(tick)`.

   - `ExecManager` updates PnL/bookkeeping, cancels stale orders, checks risk (stop-loss, drawdown), and feeds price ticks to the trading model (via `TradingStrategy`).

   - The ML model in `TradingStrategy` predicts buy/sell/hold. The output is risk-checked before a trade is sent.

   - Orders are placed with `TradeExecutor`, managed via `rest_gateway` abstraction.

   - BookKeeper updates history after trades for PnL and analytics.

3. **Backtesting/Analysis:**

   - Run `backtest_trading_strategy.py` and model scripts to train and validate strategies with historical data.

4. **Visualization:**

   - Use `visualize.py` and `live_plotter.py` for plotting and real-time monitoring.

---

## Usage

1. **Install dependencies:**

```bash

pip install -r requirements.txt

```

2. **Set up `.env` file** with your Binance testnet API keys:

```env

API_KEY=your_api_key

API_SECRET=your_api_secret

```

3. **Run main trading loop (paper/live):**

```bash

python app.py

```

4. **Backtest a strategy:**

```bash

python review_engine.py

```

5. **Visualize results:**

```bash

python visualize.py

```

---
