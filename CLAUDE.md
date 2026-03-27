# CLAUDE.md — Mistakes, Lessons & Context

> This file tracks mistakes made during development, lessons learned,
> and important context for future sessions. Updated as we go.

---

## Mistakes Log

### 2026-03-27: Session 1 — Planning & Research Phase
- **No mistakes yet in code** — planning phase only.
- **Lesson**: Deep competitive research BEFORE building saved us from
  building a clone of LLM_Trader. Identified regime detection as the
  real differentiator early.

### 2026-03-27: Build Phase — Mistake #2
- **Error**: pandas-ta 0.3.x is no longer on PyPI; 0.4.x requires Python >=3.12
- **Cause**: Planned for Python 3.11 but pandas-ta dropped support
- **Fix**: Upgraded to Python 3.13 venv, updated pyproject.toml to require >=3.12, pandas-ta>=0.4.0
- **Lesson**: Always check library Python version requirements before locking your version
- **Error**: Used `setuptools.backends._legacy:_Backend` as build backend in pyproject.toml
- **Cause**: Python 3.11's setuptools doesn't have that module path (it's a newer API)
- **Fix**: Changed to `setuptools.build_meta` (the standard, compatible backend)
- **Lesson**: Always use `setuptools.build_meta` for broad Python version compatibility

### 2026-03-27: Build Phase — Mistake #3
- **Error**: Paper broker test `test_unrealized_pnl_updates` failed with IndexError
- **Cause**: Test checked price at 48000, below the 2% stop-loss at 49000, triggering
  automatic closure and removing the position before assertion
- **Fix**: Changed test prices to stay within stop-loss range (51000 up, 49500 down)
- **Lesson**: When testing position state, account for stop-loss/take-profit triggers

### 2026-03-27: Build Phase — Mistake #4
- **Error**: LangGraph nodes wrapped in lambdas returned coroutine objects instead of dicts
- **Cause**: Lambdas can't be `async`, so `lambda state: classify_node(state, ...)` returns the coroutine without awaiting it
- **Fix**: Created proper async wrapper functions (`_make_classify_node`, etc.) instead of lambdas
- **Lesson**: Never use lambdas to wrap async functions in LangGraph nodes — always use proper async defs

### 2026-03-27: Build Phase — Mistake #5
- **Error**: Integration test mock dispatched wrong LLM response to analyze node
- **Cause**: Mock checked for "regime" keyword in user prompt, which appeared in both classify AND analyze prompts
- **Fix**: Changed mock to check system message only ("regime classifier" vs "trading analyst")
- **Lesson**: When mocking multi-call LLM flows, dispatch on system prompt (role-specific), not user content

---

## Architecture Decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Framework | LangGraph | CrewAI, raw asyncio | Graph-based flow matches our multi-step reasoning loop with conditional branching |
| LLM routing | LiteLLM | Direct API calls | Unified interface for Copilot proxy + Gemini + Ollama fallback |
| Exchange | CCXT | Direct Binance SDK | Multi-exchange support, mature library, same interface for testnet/live |
| Database | SQLite | PostgreSQL, Redis | Single-user, local-first, no infra overhead, good enough for our scale |
| Indicators | pandas-ta | TA-Lib, custom | Pure Python, no C compilation issues on Windows, covers all we need |
| Dashboard | Streamlit | PyQt6, Grafana | Fastest path to visualization, Python-native, no frontend code |
| Schema | Pydantic | dataclasses, TypedDict | Validation + serialization + structured LLM output in one tool |

## Key Context for Future Sessions

- **This is a personal-use trading agent** — no multi-tenancy, no auth, no polished UX
- **Copilot Pro+ integration** via `copilot-api` npm proxy (unofficial, could break)
- **Hybrid LLM routing**: Gemini Flash for calm markets, Claude via Copilot for complex reasoning
- **"LLM Proposes, Risk Engine Disposes"** — LLM can NEVER bypass risk limits
- **Regime detection is the CORE differentiator** — classify regime BEFORE any trade decision
- **Paper trading first** — no real money until 30+ days of proven edge
- **Build from scratch, study existing projects** — don't fork LLM_Trader/TradingAgents

## Project Structure
```
src/
├── main.py              # Entry point
├── config.py            # All settings via pydantic-settings
├── agent/               # LangGraph workflow + nodes
│   ├── graph.py         # State graph definition
│   ├── state.py         # TypedDict state
│   ├── router.py        # Model selection logic
│   ├── nodes/           # gather, classify, analyze, validate, execute, reflect, evolve
│   ├── prompts/         # Prompt templates
│   └── strategies/      # Regime-specific strategy rules
├── data/                # Market data, sentiment, narrative, indicators
├── risk/                # Deterministic risk engine
├── execution/           # Paper broker, live broker (same interface)
├── storage/             # SQLite schema + CRUD
├── llm/                 # LiteLLM wrapper + Copilot proxy config
└── dashboard/           # Streamlit UI
```
