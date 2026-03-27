# Esnaf 🤖📈

**Regime-aware autonomous crypto trading agent powered by LLM reasoning.**

Esnaf is a personal-use AI trading agent that doesn't just follow signals — it *understands* market context. Unlike traditional bots that run one static strategy, Esnaf classifies the current market regime (bull/bear/range/volatile) and selects the appropriate strategy for that regime.

## Core Innovation

```
Traditional bot:  RSI < 30 → BUY  (works 33% of the time, loses the other 67%)

Esnaf:            1. Classify regime (bull? bear? ranging?)
                  2. Select strategy for this regime
                  3. Analyze trade within regime context
                  4. Validate against hard risk limits
                  5. Execute (or sit out if uncertain)
                  6. Reflect and learn from outcomes
```

## Architecture: "LLM Proposes, Risk Engine Disposes"

- **LLM** reasons about markets, regimes, and narratives
- **Deterministic risk engine** enforces hard limits the LLM can never bypass
- **Hybrid model routing** keeps costs near-zero (~$2-5/month)

## Status

🚧 **Under active development** — Paper trading only. No real money yet.

## Quick Start

```bash
# Clone
git clone https://github.com/YEUNYUN/Esnaf.git
cd Esnaf

# Setup
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run (paper trading mode — default)
python -m src.main
```

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Agent framework | LangGraph | Graph-based flow with conditional branching |
| LLM routing | LiteLLM | Unified API for Copilot/Gemini/Ollama |
| Exchange | CCXT | Multi-exchange, testnet support |
| Indicators | pandas-ta | Pure Python, no C deps |
| Database | SQLite | Local-first, zero infra |
| Schema | Pydantic | Validation + LLM structured output |

## License

Personal use only. Not financial advice. Trade at your own risk.
