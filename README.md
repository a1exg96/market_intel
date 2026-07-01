# Market Intelligence Trading Lab v0.1

Local research/demo system for crypto market intelligence on top of Freqtrade concepts. It supports data collection, local feature engineering, baseline ML, walk-forward testing, paper trading with a virtual 1000 USD wallet, research sweeps, adaptive recommendations, and privacy auditing.

This is not financial advice and does not execute real orders.

## Model Plugin Contract

The service owns data collection, risk management, paper/live policy, execution, and audit logs. Models are replaceable plugins: each model receives the same standard input payload and returns the same standard decision payload. New model integration rules are documented in:

```text
docs/model_plugin_contract.md
```

## Install

```bash
cd market_intel_freq
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Freqtrade is optional for v0.1. The local ML/paper pipeline runs without it. Install Freqtrade separately when you want to run its backtesting CLI.

## Commands

```bash
python scripts/main.py privacy-check
python scripts/main.py db-init
python scripts/main.py collect
python scripts/main.py collector-service
python scripts/main.py features
python scripts/main.py target-audit
python scripts/main.py regime
python scripts/main.py train
python scripts/main.py audit
python scripts/main.py integrity-audit
python scripts/main.py execution
python scripts/main.py forward-paper
python scripts/main.py monte-carlo
python scripts/main.py bootstrap
python scripts/main.py benchmarks
python scripts/main.py rolling-walkforward
python scripts/main.py edge-verdict
python scripts/main.py diagnostics
python scripts/main.py rejection-report
python scripts/main.py probability-report
python scripts/main.py label-report
python scripts/main.py recommendations
python scripts/main.py paper
python scripts/main.py research
python scripts/main.py research-service
python scripts/main.py dashboard
python scripts/main.py adapt
python scripts/main.py report
python scripts/main.py full
```

`backtest` checks that `configs/config.backtest.json` is dry-run only, then calls local Freqtrade if installed:

```bash
python scripts/main.py backtest
```

## Dry-Run And Paper Trading

`configs/config.paper.json` sets `dry_run=true`, empty API keys, Telegram disabled, API server disabled, and a 1000 USDT dry-run wallet. The paper trader uses the same safety posture:

- initial balance: 1000 USD
- max risk per trade: 1 percent of balance
- `up` signal creates simulated LONG
- `down` signal creates simulated SHORT
- `flat` creates no trade

The forward paper engine is local CSV-only. It never submits exchange orders and refuses to start if `configs/paper_trading.yaml` sets `live_trading=true`.

Core terms:

- Signal: model output with symbol, direction, confidence, regime, price, model version, and executable decision.
- Open Position: a paper-only active LONG or SHORT created from an executable signal. It is not a real order.
- Closed Trade: a paper position that exited through take profit, stop loss, or fixed horizon.
- Balance: realized account value after closed paper trades only.
- Equity: balance plus unrealized PnL from open paper positions.
- Unrealized PnL: current open-position profit or loss using the latest local price.
- Realized PnL: closed-trade profit or loss already reflected in balance.

Open positions, closed trades, and signal execution audits are written locally to:

```text
data/reports/active_positions.csv
data/reports/trades.csv
data/reports/signal_execution_audit.csv
```

Dashboard endpoints:

```text
GET /api/stats
GET /api/active-positions
GET /api/trades
GET /api/signals
```

## Privacy Mode

The privacy allowlist is in:

```text
configs/privacy_policy.yaml
```

By default only market-data exchange domains are allowed. Every outbound request made through the project wrapper is logged locally:

```text
data/reports/outbound_audit.log
```

Run:

```bash
python scripts/main.py privacy-check
```

The self-test writes a blocked `example.com` request to the audit log. No logs, configs, trades, exceptions, strategies, telemetry, Sentry, Telegram, Discord, analytics, or remote logging are sent by this project.

## Optional News And On-Chain APIs

News and on-chain integrations are disabled by default in `.env.example` and `configs/privacy_policy.yaml`. To add one later, explicitly enable it in local config and add only the required domain to the allowlist. Do not commit private keys or paid API tokens.

## Outputs

- `data/raw/*.parquet`: candles and futures context
- `data/processed/*features.parquet`: engineered features
- `data/processed/predictions.parquet`: model predictions
- `data/reports/walk_forward_results.csv`: baseline validation summary
- `data/reports/feature_importance.csv`: model feature importance when available
- `data/reports/research_results.csv`: research sweep results
- `data/reports/daily_report.md`: daily local report
- `data/reports/adaptation_log.md`: adaptive recommendations
- `data/knowledge_base/*`: local knowledge base stubs

## Docker

```bash
copy .env.example .env
docker compose up --build
```

Then open:

```text
http://localhost:8000
```

The Docker setup starts a local-only demo stack:

- `postgres`: market data, signals, paper trades, reports, logs
- `redis`: local cache/pubsub events
- `collector`: recurring BTCUSDT/ETHUSDT market-data collection with synthetic fallback if APIs fail
- `research`: recurring features, targets, baseline model, paper trader, reports
- `dashboard`: FastAPI + local HTML/JS dashboard with `/health` and `/api/summary`

Useful commands:

```bash
docker compose logs -f collector
docker compose logs -f research
docker compose logs -f dashboard
docker compose down
```

Everything is local. The dashboard does not load CDN assets. No exchange API keys are required.

## Safety Notes

This project is paper-only. It refuses configs with `dry_run=false` or hardcoded exchange API keys. It is designed for local research, statistics, and strategy development.
