# AEA Crypto Trading Agent — Comprehensive Build Plan

> A personal-use, self-hosted AI agent that autonomously trades crypto by reasoning
> across price action, on-chain data, sentiment, and news — not by following fixed rules.

*Created: March 27, 2026*
*Last Updated: March 27, 2026*

---

## 0. Deep Analysis: Why Isn't Everyone Rich? (And Where The Real Opportunity Is)

### The Hard Numbers (2025-2026 Real Data)

| Statistic | Number | Source |
|---|---|---|
| Retail bot users who underperform buy-and-hold | **>80%** | CoinCentral, multiple broker disclosures |
| Retail algo traders who are net profitable | **20-45%** | Broker data, Grokipedia analysis |
| Manual day traders who are profitable | **5-10%** | Industry standard, unchanged for decades |
| Typical annualized return for successful retail algo | **5-15%** | Real broker data |
| High-Flyer (top AI quant fund) return in 2025 | **52.55%** | Fund disclosures |
| Crypto trading volume handled by bots (2026) | **60-70%** | MEXC research |

**The tools exist. The data is accessible. The APIs are available. Yet 80% of people
using trading bots STILL lose money.** This is the question worth answering.

### Why Isn't Everyone Rich? The 5 Real Reasons

#### Reason 1: Single-Strategy Fragility (The #1 Killer)

This is the biggest one. Almost every bot — including the open-source LLM agents — runs
ONE type of strategy (trend-following, mean reversion, breakout, etc.) and hopes it works.

```
The brutal reality:

  Bull market:  Trend-following works.     Mean reversion gets destroyed.
  Bear market:  Defensive/short works.     Breakout strategies get destroyed.
  Sideways:     Mean reversion works.      Trend-following gets destroyed.

  The market spends roughly equal time in each regime.
  A bot that only works in one regime will be profitable 33% of the time
  and hemorrhaging money the other 67%.
```

**This is why most bots "work" for a few weeks/months and then blow up.** They were
born in one regime and die when it changes. The existing open-source LLM bots
(LLM_Trader, MAHORAGA, etc.) all have this problem — they don't do regime detection
or strategy switching.

#### Reason 2: Alpha Decay (The Strategy Treadmill)

Any strategy that works gets discovered, copied, and arbitraged away. The more people
who run the same signals, the less those signals are worth.

- Off-the-shelf bot strategies (RSI + MACD + moving averages) are used by millions
- Open-source strategies on GitHub are readable by everyone, including institutional desks
- The edge of any published strategy approaches zero over time

**This means the value is never in the CODE. It's in the INSIGHT that drives the code.**

#### Reason 3: The Execution Gap (Backtests Lie)

- Backtests assume perfect fills at the exact price. Reality has slippage.
- Backtests don't account for the bot's own market impact.
- LLMs trained on historical data "know" what happened next (data leakage).
- A strategy showing 50% returns in backtesting often shows 5% or negative live.

**Most people never get past this stage. They see beautiful backtest results, go live,
lose money, and quit. The backtest-to-live gap is where dreams die.**

#### Reason 4: Discipline and Adaptation Failure

- People override their bots during drawdowns (panic selling)
- People crank up risk after a winning streak (greed)
- People stop monitoring and updating when life gets busy
- People don't have a systematic process for evaluating whether the bot STILL works

**The bot is only as disciplined as the human running it. And humans are not disciplined.**

#### Reason 5: It's Adversarial (Not a Factory)

**This is the deepest truth that most bot-sellers never tell you:**

Trading is not a factory that produces money if you build the right machine.
It is a zero-sum competition. For every dollar you make, someone else lost a dollar.
Your counterparties are:

- Renaissance Technologies (best quant fund in history, ~66% annual returns since 1988)
- Citadel, Two Sigma, DE Shaw — billions in infrastructure, PhDs in math
- Other retail bots running the same strategies
- Market makers with sub-millisecond execution

You are not just building a bot. You are entering a competition against the
smartest, best-funded organizations on the planet.

### So Where Is The ACTUAL Opportunity For You?

After all that research, here is what I genuinely believe is the gap. Not hype.
Real structural advantages that an LLM agent has and that most people/bots miss:

#### Gap 1: REGIME DETECTION (The Big One)

**This is what LLMs are uniquely good at and what almost no retail bot does.**

A traditional bot sees: "RSI is 25, MACD crossing up → BUY signal"
It has no idea if we're in a bull market, a bear market, or a ranging market.
It fires the same signal regardless.

An LLM can see: "RSI is 25, MACD crossing up, BUT: BTC has been in a downtrend
for 3 weeks, funding rates are deeply negative, the Fed just signaled rate hikes,
and whale wallets are moving BTC to exchanges (sell pressure). This is a bear
market oversold bounce, not a real reversal. The technical signal is a TRAP."

**No traditional bot can do this reasoning.** This is the genuine edge.

The strategy is not "buy when RSI is low." It's:
1. Detect what regime we're in (bull/bear/range/volatile)
2. Select the strategy that works in that regime
3. Only trade when the regime is clear; sit out when it's ambiguous

Most bots skip step 1 entirely. LLMs are arguably the best tool that exists
for step 1, because regime detection requires synthesizing context from
multiple domains (price, macro, sentiment, on-chain, news).

#### Gap 2: CROSS-DOMAIN SYNTHESIS (The LLM Superpower)

Traditional bots are siloed: one bot reads price, another reads sentiment,
another reads on-chain data. They don't talk to each other.

An LLM can simultaneously process:
- BTC dropped 5% (price)
- But whale wallets are accumulating, not selling (on-chain)
- SEC just approved a new crypto ETF (news)
- Fear & Greed index hit "Extreme Fear" (sentiment)
- Funding rates flipped negative — overleveraged shorts (derivatives)
- Farcaster/Lens crypto insiders are quietly bullish (niche social)

And reason: "This is forced liquidation + panic selling, not fundamental deterioration.
The smart money is buying. This is a dip-buying opportunity."

**No rules-based bot can do this synthesis. It requires understanding CONTEXT,
not just numbers.** This is what LLMs were built for.

#### Gap 3: NARRATIVE DETECTION (Pre-Move Intelligence)

Crypto is narrative-driven. Prices move because of STORIES, not fundamentals.
"AI coins," "RWA tokens," "L2 season," "DePIN" — these narratives drive
10-100x moves before fundamentals catch up.

An LLM can:
- Scan Farcaster, Lens, Crypto Twitter, Discord, DAO forums
- Detect emerging narratives BEFORE they go mainstream
- Understand that "people are talking about X the way they talked about Y
  before Y went 10x"
- Identify when a narrative is exhausted (everyone already knows → sell)

**Traditional bots can't detect narratives. They can count mentions (sentiment),
but they can't understand what those mentions MEAN.** The difference between
"everyone is talking about ETH" (bullish — new interest) and "everyone is
talking about ETH" (bearish — euphoria, top signal) requires contextual
understanding that only an LLM provides.

#### Gap 4: ADAPTIVE STRATEGY EVOLUTION

Most bots are static. You code a strategy, deploy it, and it runs the same
logic forever until it stops working.

An LLM agent can:
- Review its own performance weekly
- Identify PATTERNS in its losses ("I keep losing on Monday mornings" or
  "My trend trades fail when volatility is above X")
- Propose modifications to its own rules
- Test those modifications against recent data
- Evolve its strategy over time

**This is not science fiction — it's what Algosmithy does in Go. But nobody
has combined this with regime detection + cross-domain synthesis + narrative
detection into one coherent system.**

### The Actual Moat: It's Not Technology. It's The Loop.

The moat is not any single component. Every piece exists somewhere:
- Regime detection exists (HMM models, regime-switching bots)
- Sentiment analysis exists (Santiment, Guavy, LLM_Trader)
- On-chain analytics exists (Nansen, CryptoQuant)
- LLM trading exists (MAHORAGA, TradingAgents)
- Strategy evolution exists (Algosmithy)

**What doesn't exist: a single system that does ALL of these in a tight,
adaptive loop, tuned to one person's risk tolerance and constantly learning.**

The moat is:
1. The COMBINATION of all these signals into one reasoning engine
2. YOUR personal iteration on prompts and parameters over months
3. The agent's accumulated MEMORY of what works and what doesn't for YOU
4. The discipline of the process (paper trade → prove → scale)

This is a skill moat, not a technology moat. And skill moats are the only
moats that last in trading, because they can't be copy-pasted from GitHub.

### Existing Competitors (What's Already Built)

| Project | What It Does Well | What It's Missing |
|---|---|---|
| **LLM_Trader** | Vision, multi-exchange, CCXT, dashboard | No regime detection, no adaptive evolution, no cross-domain synthesis |
| **MAHORAGA** | Clean safety controls, multi-model | Alpaca only (equities), no on-chain, no regime detection |
| **TradingAgents** | Multi-agent debate architecture | Research-focused, no live execution, no adaptation loop |
| **Algosmithy** | Self-improving strategy loop | No LLM reasoning, no sentiment, no regime detection |
| **3Commas/Pionex** | Proven returns, years of polish | Rule-based only, no LLM reasoning, no narrative detection |

**The gap is real: nobody has built the adaptive, regime-aware, narrative-detecting,
cross-domain reasoning loop. The individual pieces exist. The integration doesn't.**

---

## 1. What We're Building (and What We're NOT)

### We ARE building:
- A **regime-aware, adaptive trading agent** — the core innovation is knowing WHEN to trade and what strategy to use, not just signal detection
- A **cross-domain reasoning engine** that synthesizes price + sentiment + on-chain + news + narrative signals simultaneously (no existing bot does all of these together)
- A **narrative-first intelligence layer** that detects proto-narratives in crypto-native social channels before they go mainstream
- A **self-evolving system** that reviews its own performance and proposes strategy modifications
- A **local-first, single-user tool** with ironclad risk management enforced at the code level
- A paper-trading-first system that proves itself before touching real money

### We are NOT building:
- A high-frequency trading bot (LLM inference is too slow; we compete on reasoning quality, not speed)
- A product for others (no auth, no multi-tenancy, no polished onboarding)
- A "set and forget" money printer (this requires tuning, monitoring, and honest evaluation)
- Something that will be profitable from day one (expect months of paper trading and iteration)
- Just another RSI + MACD bot with an LLM wrapper (that's what already exists and it doesn't work)

### Time-Limited Window (2025-2026)
This approach has a window of opportunity. LLM-based trading reasoning is currently
under-exploited because:
1. Most crypto bots are still rule-based (60-70% of crypto volume, but dumb rules)
2. Existing LLM trading agents don't do regime detection or narrative analysis
3. The tools became accessible only in the last 12 months (cheap LLM APIs, CCXT maturity)

**This window closes as more people deploy sophisticated LLM agents.** Alpha from
narrative detection will decay as LLM agents become the majority of market participants.
The goal is to build, validate, and compound returns while the edge exists.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        AEA CORE                                   │
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Data Layer   │   │  Brain       │   │  Risk Engine         │ │
│  │              │   │  (LLM Agent) │   │  (Deterministic)     │ │
│  │  - CCXT WS   │   │              │   │                      │ │
│  │  - Sentiment  │──▶│  - Analyze   │──▶│  - Position sizing   │ │
│  │  - On-chain   │   │  - Decide    │   │  - Stop-loss         │ │
│  │  - News/RSS   │   │  - Explain   │   │  - Max exposure      │ │
│  │  - Technicals │   │              │   │  - Drawdown limits   │ │
│  └──────────────┘   └──────┬───────┘   └──────────┬───────────┘ │
│                             │                      │              │
│                             ▼                      ▼              │
│                    ┌─────────────────────────────────┐           │
│                    │      Execution Layer             │           │
│                    │  (CCXT — testnet/live toggle)    │           │
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

### The Critical Design Principle: LLM Proposes, Risk Engine Disposes

The LLM is the "brain" — it reasons about what to do. But it CANNOT directly execute trades.
Every LLM decision passes through a deterministic risk engine written in plain Python with
hard-coded limits. The LLM cannot override, circumvent, or modify these limits.

```
LLM output: "Buy 2 BTC at market"
     │
     ▼
Risk Engine checks:
  ├── Does this exceed max position size? → REJECT
  ├── Does this exceed per-trade budget? → REJECT
  ├── Does this exceed portfolio exposure limit? → REJECT
  ├── Are we in daily drawdown pause? → REJECT
  ├── Is this a duplicate/conflicting order? → REJECT
  └── All checks pass → EXECUTE via CCXT
```

---

## 3. Tech Stack (Decided, With Justification)

| Component | Choice | Why |
|---|---|---|
| **Language** | Python 3.12+ | Ecosystem dominance for AI + trading. CCXT, LangGraph, all data libs are Python-first. |
| **Agent Framework** | LangGraph | Graph-based state machine with checkpointing. Crash recovery, audit trails, conditional workflows. Steeper learning curve but correct for financial automation where reliability matters. |
| **LLM — Primary** | Claude Sonnet via Copilot Pro+ | Uses your existing subscription. Zero marginal cost within 1,500 premium requests/mo. Best reasoning-to-cost ratio. |
| **LLM — Fast/Cheap** | Gemini 3 Flash (direct API) | $0.10-0.50/$0.30-3.00 per 1M tokens. Use for simple classification tasks (sentiment score, trend direction). Handles ~70% of cycles to preserve Copilot quota. |
| **LLM — Local Fallback** | Ollama + DeepSeek R1 8B or Llama 3.3 8B | Zero API cost. Use when Copilot quota is exhausted or API is down. ~6-10 tokens/sec on decent hardware. |
| **LLM Router** | LiteLLM | Unified API across all providers. Route Copilot proxy, Gemini, Ollama through one interface. Handles retries, fallbacks, cost tracking. |
| **Copilot Bridge** | copilot-api (npm proxy) | Exposes Copilot Pro+ models as OpenAI-compatible API on localhost. LiteLLM connects to it as if it were OpenAI. |
| **Exchange Interface** | CCXT + CCXT Pro (WebSocket) | Unified API for 100+ exchanges. Testnet support built in. Async support. The standard. |
| **Backtesting** | VectorBT | Blazing fast vectorized backtesting. Native CCXT integration. ML signal support. |
| **Database** | SQLite | Local-first. Zero setup. Perfect for single-user. |
| **Scheduling** | APScheduler | Lightweight, in-process. No Celery overhead for single-user. |
| **Dashboard** | Streamlit | Fast to build, good enough for personal use. |

### LLM Cost Strategy: Hybrid Model Routing

```
Every 15min cycle → Router decides which model to use:

IF market is calm (volatility < threshold) AND no open positions:
  → Gemini Flash (near-free, just checks "anything happening?")
  → Does NOT consume Copilot quota

IF market is active OR position is open OR potential trade signal:
  → Claude Sonnet via Copilot Pro+ (deep reasoning)
  → Consumes 1 premium request

IF Copilot quota exhausted (>1,500/mo):
  → Gemini 3.1 Pro (paid, ~$2-3/1M tokens, still cheap)
  → OR Ollama local fallback (free but slower reasoning)

Estimated monthly Copilot usage with this strategy:
  ~30% of 2,880 cycles = ~864 premium requests/mo
  Well within the 1,500 limit. No overages needed.
```

### What About MCP / OpenClaw / Claude Computer Use?

After research, honest assessment:

- **MCP**: LangGraph already provides tool calling natively. MCP adds abstraction we don't need.
- **OpenClaw**: General-purpose assistant framework. Our agent needs a tight trading loop, not a "do anything" platform.
- **Claude Computer Use**: We're calling APIs, not clicking buttons. Irrelevant here.

**Use the right tool for the job, not the newest shiny thing.**

---

## 4. The Agent's Reasoning Loop (Regime-First Architecture)

Every N minutes (configurable, default 15min), the agent runs this cycle.
The key innovation vs existing bots: **regime detection happens FIRST, before
any trade decision.** This is the step that 99% of bots skip.

### Step 1: GATHER (Data Collection Node)
```
Inputs collected:
  - Current price + OHLCV candles (CCXT WebSocket)
  - Order book depth (top 10 bids/asks)
  - Technical indicators (RSI, MACD, Bollinger Bands, volume profile)
    → Computed locally, NOT by the LLM (deterministic, free, fast)
  - Sentiment score (Reddit mentions, news sentiment, fear/greed index)
  - Narrative signals (Farcaster, Lens, Crypto Twitter — emerging themes)
  - On-chain metrics (exchange inflows/outflows, whale movements)
  - Current portfolio state (open positions, unrealized P&L, available capital)
  - Recent trade history (what did the agent do in the last 24h and why)
  - Agent's memory (lessons learned from past mistakes/successes)
```

### Step 2: ROUTE (Model Selection — Deterministic, No LLM)
```python
def select_model(market_data, portfolio):
    volatility = compute_volatility(market_data)
    has_positions = len(portfolio.open_positions) > 0
    has_signal = any_signal_detected(market_data)

    if volatility < CALM_THRESHOLD and not has_positions and not has_signal:
        return "gemini-flash"      # Cheap check, preserves Copilot quota
    else:
        return "copilot-claude"    # Deep reasoning needed
```

### Step 3: CLASSIFY REGIME (The Core Innovation — LLM Node)
```
BEFORE any trade decision, the LLM classifies the current market regime.
This is what separates us from every open-source LLM trading bot.

Input: All gathered data from Step 1
Output:
{
  "regime": "bull_trend | bear_trend | ranging | high_volatility | regime_shift",
  "regime_confidence": 0.0-1.0,
  "regime_reasoning": "3-4 sentences explaining why this regime",
  "active_narratives": ["AI tokens", "L2 season", "BTC ETF flows"],
  "narrative_stage": "early | mainstream | exhausted",
  "recommended_strategy": "trend_follow | mean_revert | defensive | sit_out",
  "strategy_reasoning": "Why this strategy fits this regime"
}

CRITICAL: If regime_confidence < 0.6, the agent defaults to SIT OUT.
Ambiguous regimes are the most dangerous. Doing nothing is a valid strategy.
```

### Step 4: ANALYZE (Trade Decision — LLM Node)
```
Only runs if Step 3 recommends a strategy other than "sit_out."
The LLM makes a specific trade decision WITHIN the classified regime.

Input: All gathered data + regime classification + selected strategy
Output:
{
  "action": "BUY | SELL | HOLD | CLOSE_LONG | CLOSE_SHORT",
  "asset": "BTC/USDT",
  "confidence": 0.0-1.0,
  "size_suggestion": "small | medium | large",
  "reasoning": "2-3 sentences explaining the decision",
  "timeframe": "15m | 1h | 4h | 1d",
  "stop_loss_pct": 2.0,
  "take_profit_pct": 5.0,
  "key_factors": ["whale accumulation", "RSI oversold", "narrative momentum"],
  "regime_alignment": "How this trade aligns with the classified regime"
}

CRITICAL: Temperature = 0. Structured output enforced via tool calling.
The LLM MUST output valid JSON matching the schema or the action is rejected.
```

### Step 5: VALIDATE (Risk Engine Node — Deterministic, No LLM)
```python
# This is plain Python. The LLM cannot influence, modify, or bypass this.
def validate_trade(proposal, portfolio, risk_config, regime):
    if portfolio.daily_drawdown >= risk_config.max_daily_drawdown:
        return REJECT("Daily drawdown limit hit. All trading paused.")
    if regime.confidence < 0.6:
        return REJECT("Regime unclear. Sitting out.")
    if proposal.position_value > risk_config.max_per_trade:
        return REJECT(f"Trade size ${proposal.value} exceeds max ${risk_config.max_per_trade}")
    if portfolio.total_exposure + proposal.value > risk_config.max_total_exposure:
        return REJECT("Would exceed total portfolio exposure limit")
    if proposal.confidence < risk_config.min_confidence_threshold:
        return REJECT(f"Confidence {proposal.confidence} below threshold")
    if similar_trade_in_last_n_minutes(proposal, cooldown=30):
        return REJECT("Cooldown period — similar trade too recent")
    # Regime-specific validation: higher bar for counter-trend trades
    if regime.type == "bear_trend" and proposal.action == "BUY":
        if proposal.confidence < 0.8:
            return REJECT("Counter-trend trade needs >0.8 confidence in bear regime")
    return APPROVE(proposal)
```

### Step 6: EXECUTE (Execution Node)
```
If approved → Place order via CCXT (testnet or live, same code)
If rejected → Log reason, continue monitoring
All executions are logged with full context (regime, data seen, decision, reasoning)
```

### Step 7: REFLECT (Post-Trade Analysis — runs hourly)
```
The LLM reviews its recent decisions:
  - Which trades are in profit? Which aren't?
  - Was the regime classification accurate in hindsight?
  - Did the narrative signals prove predictive or misleading?
  - Are there patterns in its mistakes?

Produces a structured "lessons learned" entry stored in memory.
```

### Step 8: EVOLVE (Weekly Strategy Review)
```
The LLM reviews the entire week's performance:
  - Win/loss by regime type (are we good at trending but bad at ranging?)
  - Win/loss by strategy (which approaches are working?)
  - Narrative prediction accuracy (did we correctly spot early/exhausted narratives?)
  - Proposes specific adjustments:
    - "Increase confidence threshold for mean-reversion trades"
    - "Add weight to Farcaster sentiment — it led BTC moves by 4h twice this week"
    - "Reduce position size in volatile regime — drawdowns are too large"
  
  These proposals are logged for human review. NOT auto-applied.
  You decide which adjustments to make. The agent suggests, you approve.
```

---

## 5. Data Sources

### Free Tier (Start Here)
| Data Type | Source | Cost | Limits |
|---|---|---|---|
| Price data (real-time) | CCXT Pro WebSocket via Binance/Bybit | Free | Rate limits only |
| Price data (historical) | CCXT REST | Free | Rate limits |
| Technical indicators | Computed locally (pandas-ta) | Free | None |
| Crypto news | CryptoPanic API | Free | 5 req/min |
| Reddit sentiment | Reddit API (free tier) | Free | 100 req/min |
| Fear & Greed Index | Alternative.me API | Free | Reasonable |
| Market overview | CoinGecko API | Free | 30 req/min |

### Paid Tier (Only If Profitable)
| Data Type | Source | Cost | Value |
|---|---|---|---|
| Pro sentiment | Santiment API | ~$49/mo | Deep social + on-chain |
| AI-native sentiment | Guavy API | ~$29/mo | Real-time narrative detection |
| Whale tracking | Nansen | ~$100/mo | Smart money flow analysis |
| Deep on-chain | CryptoQuant | ~$39/mo | Exchange flows, miner activity |

**Rule: Don't pay for data until the agent is consistently profitable on free data.**

---

## 6. Risk Management Configuration

```python
RISK_CONFIG = {
    # Capital limits
    "max_total_capital": 1000.00,
    "max_per_trade": 50.00,
    "max_open_positions": 5,
    "max_exposure_pct": 0.30,          # Max 30% deployed at once

    # Loss limits
    "stop_loss_pct": 3.0,
    "max_daily_drawdown": 50.00,
    "max_weekly_drawdown": 100.00,

    # Agent behavior limits
    "min_confidence_threshold": 0.65,
    "trade_cooldown_minutes": 15,
    "max_trades_per_day": 20,

    # Emergency
    "kill_switch": False,
    "human_approval_above": 200.00,
}
```

**Enforced in deterministic code, not in the LLM prompt.**

---

## 7. Paper Trading Strategy

### Phase 1: Backtesting (Week 1-2)
- Download 6+ months BTC/USDT and ETH/USDT historical data
- Run agent's reasoning loop against historical data via VectorBT
- Measure: Win rate, Sharpe ratio, max drawdown, profit factor
- Compare against: buy-and-hold, simple MA crossover, random

### Phase 2: Local Paper Trading (Week 3-6)
- Agent runs against LIVE market data, local simulated broker
- Tracks hypothetical P&L in SQLite
- All decisions logged with full reasoning

### Phase 3: Exchange Testnet (Week 7-8)
- Move to Binance testnet via CCXT (same code, different endpoint)
- Validates order execution logic, fills, error handling
- Note: Testnet liquidity is unrealistic; data resets monthly

### Phase 4: Micro-Live (After Proven Edge)
- ONLY after 30+ days of positive paper trading
- Start with $50-100 real money
- Same risk limits, same agent, just real execution

### What "Proven Edge" Means
```
Minimum criteria before going live:
  ├── Paper trading P&L positive over 30+ days
  ├── Sharpe ratio > 1.0
  ├── Win rate > 40% with avg win > avg loss
  ├── Max drawdown < 15% of capital
  ├── Outperforms buy-and-hold BTC in same period
  └── At least 50 trades executed (statistical significance)

If ANY of these fail → keep paper trading and iterating.
```

---

## 8. Cost Analysis

### Monthly Costs (Estimated)
```
LLM APIs:
  Copilot Pro+: $0 extra (already paying for subscription)
  Gemini Flash (70% of cycles): ~$2-5/month
  Ollama local (fallback): $0
  ────────────────────
  Total LLM: ~$2-5/month

Data APIs: $0 (free tier to start)
Electricity: negligible
────────────────────
Total: ~$2-5/month (!)
```

**This is a massive improvement over the $35-57/month estimated before the Copilot
integration. The break-even bar is now much lower — the agent needs to generate
only ~$5/month (~0.5% on $1000) to cover costs.**

### If Copilot Quota Is Exceeded
```
Overage at $0.04/request:
  ~200 extra requests = $8/month
  Still far cheaper than direct API keys.

OR: Fall back to Gemini 3.1 Pro for the remainder of the month
  ~$10-15 extra. Still cheaper than Anthropic direct.
```

---

## 9. Risks — Complete Honest Assessment

### Technical Risks
| Risk | Severity | Mitigation |
|---|---|---|
| LLM hallucination on trade decisions | HIGH | Structured output + risk engine. Agent can never act on raw text. |
| LLM inconsistency | MEDIUM | Temperature 0, structured output, decision logging |
| Copilot API proxy breaks (unofficial tool) | MEDIUM | LiteLLM auto-fallback to Gemini → Ollama. Agent holds if all fail. |
| Exchange API downtime | MEDIUM | CCXT error handling, retry logic, fail-safe to HOLD |
| Bug in risk engine | CRITICAL | Extensive unit tests. <200 lines of simple Python. |
| State corruption / crash | LOW | LangGraph checkpointing. Resumes from last checkpoint. |

### Financial Risks
| Risk | Severity | Mitigation |
|---|---|---|
| Agent loses money consistently | HIGH | Paper trade first. Kill switch. Weekly review. |
| Flash crash / black swan | HIGH | Hard stop-loss. Daily drawdown limit halts all trading. |
| Overtrading | MEDIUM | Max trades/day. Cooldown periods. Fee tracking. |
| Fees eroding profits | MEDIUM | Factor 0.1% per trade into all P&L. Paper trading simulates fees. |

### LLM-Specific Risks
| Risk | Severity | Reality Check |
|---|---|---|
| LLMs are not proven alpha generators | HIGH | No public evidence of consistent market-beating. This is experimental. |
| Confident wrong answers | HIGH | Risk engine is the real gatekeeper, not the LLM's self-reported confidence. |
| Model updates change behavior | MEDIUM | Pin model versions. Test on paper before updating. |
| Prompt injection via data | LOW | Sanitize all external data before including in prompts. |

### Legal Risks
| Risk | Details |
|---|---|
| Legality | Personal crypto bot trading is **legal** in US, EU, UK, and most jurisdictions (2026). |
| Market manipulation | Don't spoof, wash trade, or manipulate order books. Even accidentally. |
| Tax obligations | Every trade is a taxable event. Log everything. |
| Exchange ToS | Binance, Bybit APIs **allow** bot trading. That's what the APIs are for. |
| Copilot ToS | Using copilot-api proxy is unofficial. GitHub could restrict this. Low risk but real. Risk mitigated by having Gemini/Ollama fallbacks. |

---

## 10. Development Phases

### Phase 1: Foundation (Core Trading Loop)
- Project setup (Python, venv, dependencies)
- copilot-api proxy setup + LiteLLM configuration
- CCXT integration (Binance testnet, fetch prices, place test orders)
- Technical indicator computation (RSI, MACD, BBands via pandas-ta)
- Risk engine (all hard limits, position sizing, stop-loss enforcement)
- SQLite schema (trades, portfolio, decisions, regime log, P&L)
- LangGraph agent skeleton (gather → route → classify regime → analyze → validate → execute)
- LLM integration via LiteLLM (Copilot primary, Gemini cheap, Ollama fallback)
- Regime classification node (the core differentiator — gets built early)
- Local paper trading broker (simulated execution against live prices)
- Basic CLI output (state, trades, P&L, current regime)

### Phase 2: Intelligence (Make It Smart)
- Sentiment data pipeline (Reddit, CryptoPanic, Fear & Greed)
- Narrative detection pipeline (Farcaster, Lens, Crypto Twitter scanning)
- News ingestion and LLM-powered summarization
- On-chain data integration (exchange flows, whale tracking)
- Agent memory system (lessons learned, pattern recognition, regime history)
- Strategy library (trend-follow, mean-revert, defensive, sit-out)
- Strategy-regime mapping (which strategy to use in which regime)
- Prompt engineering and optimization
- Backtesting pipeline with VectorBT
- Decision logging and replay (audit trail)

### Phase 3: Validation (Prove It Works)
- Run 30+ day paper trading campaign
- Performance analytics (Sharpe, Sortino, win rate, drawdown)
- Regime classification accuracy tracking (was the regime call correct?)
- Compare vs benchmarks (buy-and-hold, MA crossover, single-strategy bots)
- Cost tracking (API spend vs profits)
- Weekly EVOLVE loop (agent proposes adjustments, human approves)
- Iterate on prompts, risk params, data sources
- Streamlit dashboard (with regime timeline visualization)

### Phase 4: Live Trading (Only If Phase 3 Succeeds)
- Binance testnet validation (real order flow)
- Micro-live with $50-100
- Real vs paper performance comparison
- Tax logging integration
- Telegram notifications

---

## 11. File Structure

```
aea/
├── main.py                  # Entry point
├── config.py                # All configuration (risk limits, API keys, pairs, intervals)
├── requirements.txt
├── .env                     # API keys (gitignored)
│
├── agent/
│   ├── graph.py             # LangGraph workflow definition
│   ├── router.py            # Model selection logic (which LLM for this cycle)
│   ├── nodes/
│   │   ├── gather.py        # Data collection node
│   │   ├── classify.py      # Regime classification node (CORE INNOVATION)
│   │   ├── analyze.py       # Trade decision node (within classified regime)
│   │   ├── validate.py      # Risk engine node
│   │   ├── execute.py       # Order execution node
│   │   ├── reflect.py       # Post-trade reflection node (hourly)
│   │   └── evolve.py        # Weekly strategy evolution node
│   ├── prompts/
│   │   ├── regime.py        # Regime classification prompt templates
│   │   ├── analysis.py      # Trading analysis prompt templates
│   │   ├── reflection.py    # Self-reflection prompt templates
│   │   └── evolution.py     # Strategy evolution prompt templates
│   ├── strategies/
│   │   ├── base.py          # Strategy interface
│   │   ├── trend_follow.py  # Trend-following strategy rules
│   │   ├── mean_revert.py   # Mean reversion strategy rules
│   │   └── defensive.py     # Bear market / capital preservation rules
│   └── state.py             # LangGraph state definition (TypedDict)
│
├── data/
│   ├── market.py            # Price data via CCXT
│   ├── sentiment.py         # Reddit, news, fear/greed
│   ├── narrative.py         # Farcaster, Lens, Crypto Twitter narrative detection
│   ├── onchain.py           # On-chain data sources
│   └── indicators.py        # Technical indicator computation
│
├── risk/
│   ├── engine.py            # Core risk validation (deterministic)
│   ├── position.py          # Position sizing
│   └── limits.py            # Hard limit definitions
│
├── execution/
│   ├── broker.py            # Abstract broker interface
│   ├── paper_broker.py      # Local paper trading broker
│   ├── live_broker.py       # CCXT live/testnet execution
│   └── order_types.py       # Order models
│
├── storage/
│   ├── database.py          # SQLite connection and schema
│   ├── trades.py            # Trade history CRUD
│   ├── decisions.py         # Agent decision log CRUD
│   └── portfolio.py         # Portfolio state tracking
│
├── dashboard/
│   └── app.py               # Streamlit dashboard
│
├── backtest/
│   └── runner.py            # VectorBT backtesting pipeline
│
└── tests/
    ├── test_risk_engine.py  # CRITICAL — most important tests
    ├── test_execution.py
    ├── test_indicators.py
    └── test_agent_flow.py
```

---

## 12. Key Dependencies

```
# Core
langchain-core>=0.3
langgraph>=0.3
litellm>=1.50

# Exchange
ccxt>=4.4

# Data & Analysis
pandas>=2.2
numpy>=2.0
pandas-ta>=0.3
vectorbt>=0.26

# LLM Providers
google-generativeai>=0.8   # Gemini Flash (direct, cheap)

# Web & Scheduling
streamlit>=1.40
apscheduler>=3.10

# Utilities
pydantic>=2.9
python-dotenv>=1.0
httpx>=0.27
loguru>=0.7
openai>=1.50              # For copilot-api proxy (OpenAI-compatible)
```

---

## 13. Honest Summary (Updated After Deep Research)

**The answer to "why isn't everyone rich?":**
- 80%+ of bot users underperform buy-and-hold. The tools aren't the bottleneck.
- Bots fail because they run ONE static strategy in a market that changes regime constantly.
- Alpha decays: any published strategy gets arbitraged away.
- Backtests lie: LLMs leak historical data, slippage isn't modeled, overfitting is rampant.
- Trading is adversarial: for every winner there's a loser, and your counterparties have billions.

**Where we genuinely differentiate from what exists:**
- **Regime detection as the CORE feature** — no existing open-source LLM bot does this
- **Cross-domain synthesis** — connecting on-chain + sentiment + narrative + price in one reasoning engine
- **Narrative-first intelligence** — detecting proto-narratives before mainstream
- **Self-evolving strategy loop** — weekly review and adaptation proposals
- **The COMBINATION of all these** is what doesn't exist. Individual pieces do.

**What's realistic:**
- Building a working paper-trading agent: 4-6 weeks
- Building one that doesn't lose money in paper trading: 2-4 months
- Building one that reliably profits above costs: Unknown. Maybe never.

**What's in your favor:**
- Copilot Pro+ integration makes LLM costs near-zero (~$2-5/mo vs $35-57/mo)
- Crypto markets are less efficient than stocks — room for information edge
- Regime detection is a genuine, under-exploited approach
- The time window (2025-2026) is real — LLM trading is still early
- You learn an enormous amount even if it never profits
- Paper trading means you learn cheaply

**What's against you:**
- Competing against institutional quant firms with billions in infrastructure
- LLMs will sound confident when wrong — don't confuse articulate reasoning with correct reasoning
- Past backtesting success ≠ live success (the #1 reason for false hope)
- This time window will close as LLM agents become commodity
- Alpha from any specific approach decays once others discover it

**The moat is a skill moat, not a technology moat:**
Your edge builds over months of: tuning prompts, understanding which data sources
actually predict price moves, developing regime classification accuracy, and building
the agent's memory of what works for YOUR specific trading style. This can't be
copy-pasted from GitHub. It's earned through iteration and discipline.

**The right mindset:**
This is an experiment at the frontier of what LLMs can do in finance. The near-zero
LLM cost via Copilot makes the economics viable. Budget for it like a hobby ($2-5/mo).
Go in expecting to learn, not to profit. If profit comes, it's because you earned it
through months of disciplined iteration — not because you built the right code.
