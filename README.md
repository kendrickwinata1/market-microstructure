# Market Microstructure Trading Engine

## Overview

This project is a modular Python framework for developing, backtesting, and running live trading strategies for cryptocurrency (specifically BTC/USDT) on Binance Futures (testnet).

The architecture integrates live data streaming, REST API execution, order management, ML-based signal generation, risk management, bookkeeping, backtesting, and visualization tools.

---

## UML Diagram
![UML Diagram](https://github.com/kendrickwinata1/market-microstructure/blob/main/UML.png?raw=true)

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
