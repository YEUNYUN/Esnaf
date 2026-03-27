# Esnaf 🤖📈

**Regime-aware autonomous crypto trading agent powered by LLM reasoning.**

---

Esnaf is a personal-use AI trading agent that doesn't just follow signals — it *understands* market context. Unlike traditional bots that run a single strategy and hope for the best, Esnaf classifies the current market regime (bull, bear, range, or volatile) **before** making any trading decision, then selects the appropriate strategy for that regime.

The core insight is simple: a trend-following strategy that prints money in a bull market will hemorrhage capital in a sideways range. Most bots ignore this entirely. Esnaf's regime-first architecture means it knows *when* to trade, *what* strategy to use, and — critically — *when to sit out* because the market is ambiguous.

The system follows a strict **"LLM Proposes, Risk Engine Disposes"** architecture. The LLM reasons about markets, regimes, and narratives, but every decision must pass through a deterministic risk engine written in plain Python with hard-coded limits. The LLM cannot override, circumvent, or modify these limits — ever. This separation keeps the creative reasoning of LLMs while preventing the catastrophic outcomes that unconstrained AI trading could produce.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ESNAF CORE                                │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Data Layer   │   │  Brain       │   │  Risk Engine         │ │
│  │              │   │  (LLM Agent) │   │  (Deterministic)     │ │
│  │  - CCXT      │   │              │   │                      │ │
│  │  - Sentiment  │──▶│  - Classify  │──▶│  - Position sizing   │ │
│  │  - News/RSS   │   │  - Analyze   │   │  - Stop-loss         │ │
│  │  - Technicals │   │  - Explain   │   │  - Max exposure      │ │
│  │              │   │              │   │  - Drawdown limits   │ │
│  └──────────────┘   └──────┬───────┘   └──────────┬───────────┘ │
│                             │                      │             │
│                             ▼                      ▼             │
│                    ┌─────────────────────────────────┐           │
│                    │      Execution Layer             │           │
│                    │  Paper broker / CCXT live/testnet│           │
│                    └─────────────────────────────────┘           │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────────────────────┐           │
│                    │      State & Memory              │           │
│                    │  SQLite + LangGraph checkpoints  │           │
│                    └─────────────────────────────────┘           │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────────────────────┐           │
│                    │      Dashboard (Streamlit)       │           │
│                    └─────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Regime-first trading** — classifies market as bull/bear/range/volatile *before* any trade decision
- **LLM proposes, risk engine disposes** — creative AI reasoning + deterministic hard limits
- **Hybrid model routing** — Claude Sonnet for deep reasoning, Gemini Flash for calm markets, Ollama fallback (~$2–5/month)
- **Paper trading by default** — must explicitly prove an edge before risking real money
- **Cross-domain synthesis** — combines price action, technicals, sentiment, and news in a single reasoning step
- **Persistent memory** — agent learns from past trades and carries lessons across restarts
- **Self-reflection** — reviews its own decisions and identifies patterns in mistakes
- **Kill switch** — automatic trading halt if drawdown exceeds safety threshold
- **Full audit trail** — every decision logged with regime, reasoning, and confidence

## Quick Start

### Prerequisites

- **Python 3.13+**
- **Node.js** (for the Copilot API proxy, optional)
- **Ollama** (for local LLM fallback, optional)

### Install

```bash
git clone https://github.com/YEUNYUN/Esnaf.git
cd Esnaf

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys — see Configuration section below
```

### Run (Paper Trading)

```bash
python -m src.main
```

The agent will start in paper trading mode by default — no real money is at risk.

## Project Structure

```
esnaf/
├── src/
│   ├── main.py                 # Entry point — initializes components, runs trading loop
│   ├── config.py               # All settings via pydantic-settings (loaded from .env)
│   │
│   ├── agent/                  # LangGraph workflow orchestrator
│   │   ├── graph.py            # Builds the 7-node reasoning graph
│   │   ├── state.py            # TypedDict state, enums (Regime, Strategy, Action)
│   │   ├── router.py           # Deterministic model selection (cheap vs deep)
│   │   ├── memory.py           # Persistent lessons learned across sessions
│   │   ├── nodes/
│   │   │   ├── gather.py       # Fetch market data + compute indicators
│   │   │   ├── classify.py     # Regime classification (the core innovation)
│   │   │   ├── analyze.py      # Trade decision within classified regime
│   │   │   ├── validate.py     # Risk engine gate (deterministic, unbypassable)
│   │   │   ├── execute.py      # Order execution via broker
│   │   │   └── reflect.py      # Post-trade reflection + memory storage
│   │   ├── prompts/            # LLM prompt templates
│   │   └── strategies/         # Strategy definitions per regime
│   │
│   ├── data/                   # Market data & sentiment pipeline
│   │   ├── market.py           # CCXT-based OHLCV + price fetching
│   │   ├── indicators.py       # RSI, MACD, Bollinger Bands via pandas-ta
│   │   ├── sentiment.py        # Fear & Greed, CryptoPanic, CoinGecko
│   │   └── narrative.py        # Emerging narrative detection
│   │
│   ├── risk/                   # Deterministic risk management
│   │   └── engine.py           # Hard limits, kill switch, position validation
│   │
│   ├── llm/                    # LLM integration
│   │   └── client.py           # LiteLLM routing (Copilot → Gemini → Ollama)
│   │
│   ├── execution/              # Trade execution
│   │   └── paper_broker.py     # Simulated broker for paper trading
│   │
│   ├── storage/                # Persistence
│   │   └── database.py         # SQLite: trades, decisions, portfolio, memory
│   │
│   ├── backtest/               # Historical backtesting (VectorBT)
│   └── dashboard/
│       └── app.py              # Streamlit monitoring dashboard
│
├── tests/
│   ├── unit/                   # Unit tests for each module
│   └── integration/            # End-to-end cycle tests
│
├── .env.example                # All configuration with documentation
├── pyproject.toml              # Dependencies and tool config
└── conftest.py                 # Shared test fixtures
```

## How It Works — The 7-Node Reasoning Loop

Every cycle (default: 15 minutes), the agent runs through this graph:

### 1. GATHER — Data Collection
Fetches OHLCV candles via CCXT, computes technical indicators (RSI, MACD, Bollinger Bands, ATR) locally with pandas-ta, and pulls sentiment data from free APIs (Fear & Greed Index, CryptoPanic news, CoinGecko).

### 2. ROUTE — Model Selection (Deterministic)
Selects which LLM to use based on market conditions. Calm markets with no open positions → Gemini Flash (near-free). Active markets or open positions → Claude Sonnet via Copilot proxy (deep reasoning). This keeps monthly LLM costs at ~$2–5.

### 3. CLASSIFY — Regime Detection (The Core Innovation)
**Before any trade decision**, the LLM classifies the current market regime: `BULL_TREND`, `BEAR_TREND`, `RANGING`, `HIGH_VOLATILITY`, or `REGIME_SHIFT`. It also recommends a strategy (`TREND_FOLLOW`, `MEAN_REVERT`, `DEFENSIVE`, or `SIT_OUT`). If regime confidence is below 0.6, the agent sits out — doing nothing is a valid strategy.

### 4. ANALYZE — Trade Decision
Only runs if the regime recommends a strategy other than sit-out. The LLM proposes a specific action (BUY, SELL, HOLD, CLOSE_LONG, CLOSE_SHORT) with confidence score, reasoning, and suggested position size — all within the context of the classified regime.

### 5. VALIDATE — Risk Engine Gate
The deterministic risk engine checks the proposal against hard limits: max position size, daily drawdown, total exposure, confidence threshold, trade cooldown, and kill switch. **The LLM cannot override this step.** Counter-trend trades (e.g., buying in a bear market) require higher confidence (0.8+).

### 6. EXECUTE — Order Execution
If validated, the trade is executed via the paper broker (or CCXT for live/testnet). Position tracking, stop-loss, and take-profit are managed automatically.

### 7. REFLECT — Post-Trade Analysis
The agent reviews the outcome, stores lessons learned in persistent memory, and logs the full decision context to SQLite for audit and future learning.

## Risk Management

The risk engine is the agent's safety net — deterministic Python code that the LLM can never influence:

| Limit | Default | Purpose |
|---|---|---|
| Max capital | $10,000 | Total capital under management |
| Max position size | 10% | No single position exceeds this % of capital |
| Max daily drawdown | 3% | All trading paused if hit |
| Max total exposure | 30% | Max % of capital deployed at once |
| Min confidence | 0.7 | LLM must be at least this confident |
| Min regime confidence | 0.6 | Below this → sit out entirely |
| Counter-trend confidence | 0.8 | Higher bar for trades against the regime |
| Trade cooldown | 30 min | No duplicate trades within this window |
| Max trades/day | 10 | Prevents overtrading |
| Stop-loss | 3% | Per-position automatic exit |
| Kill switch | 5% | Emergency halt if drawdown exceeds this |
| Fee rate | 0.1% | Simulated in all P&L calculations |

All limits are configurable via `.env` but enforced in code — not in prompts.

## Paper Trading

Paper trading is the **default mode**. The agent trades against live market data but uses a simulated broker:

```bash
# Just run it — paper trading is on by default
python -m src.main
```

The paper broker:
- Fills orders at the current market price
- Tracks open positions with entry price, quantity, and P&L
- Enforces stop-loss and take-profit levels
- Simulates exchange fees (0.1% per trade)
- Persists portfolio state in SQLite

**Do not switch to live trading until you have:**
- 30+ days of positive paper trading
- Sharpe ratio > 1.0
- Win rate > 40% with average win > average loss
- Max drawdown < 15% of capital
- Outperformance vs buy-and-hold BTC in the same period

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

## Dashboard

Launch the Streamlit monitoring dashboard:

```bash
pip install -e ".[dashboard]"
streamlit run src/dashboard/app.py
```

The dashboard reads from the SQLite database (read-only) and displays:
- Trade history and decision log
- Portfolio equity curve
- Regime classification timeline
- P&L breakdown

## Configuration

All settings are managed through environment variables loaded from `.env`. See [`.env.example`](.env.example) for a complete reference with documentation for every setting.

Key sections:
- **Exchange** — Binance API keys, testnet toggle
- **LLM** — Model identifiers, Copilot proxy URL, Gemini API key
- **Risk** — All hard limits (capital, position size, drawdown, kill switch)
- **Agent** — Trading pair, cycle interval, log level, paper trading toggle
- **Sentiment** — CryptoPanic API key for news headlines

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Agent framework | LangGraph | Graph-based flow with conditional branching |
| LLM routing | LiteLLM | Unified API for Copilot/Gemini/Ollama |
| Exchange | CCXT | Multi-exchange, testnet support |
| Indicators | pandas-ta | Pure Python, no C dependencies |
| Database | SQLite (aiosqlite) | Local-first, zero infrastructure |
| Schema | Pydantic | Validation + LLM structured output |
| Dashboard | Streamlit + Plotly | Quick personal monitoring UI |
| Config | pydantic-settings | Type-safe, .env-driven configuration |

## ⚠️ Disclaimer

**This software is NOT financial advice.**

Esnaf is a personal experimental tool for exploring AI-assisted trading. It is provided as-is, with no guarantees of any kind.

- **Past performance does not predict future results.** Paper trading profits do not guarantee live trading profits.
- **You can lose money.** Crypto markets are volatile and unpredictable. Automated trading adds additional risks including software bugs, API failures, and model hallucinations.
- **This is a personal tool, not a product.** It was built for one person's use. There is no support, no warranty, and no guarantee that it works correctly.
- **LLMs are not proven alpha generators.** There is no public evidence that LLM-based trading consistently beats the market. This is an experiment.
- **Use at your own risk.** The author is not responsible for any financial losses incurred from using this software.

If you choose to trade with real money, start with an amount you can afford to lose entirely.

## License

MIT
