"""Agent state definition for the LangGraph workflow.

This TypedDict flows through every node in the graph. Each node reads
what it needs and writes its outputs. LangGraph handles persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel

# === Enums ===


class Regime(StrEnum):
    """Market regime classification."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    REGIME_SHIFT = "regime_shift"
    UNKNOWN = "unknown"


class Strategy(StrEnum):
    """Trading strategy selected based on regime."""

    TREND_FOLLOW = "trend_follow"
    MEAN_REVERT = "mean_revert"
    DEFENSIVE = "defensive"
    SIT_OUT = "sit_out"


class Action(StrEnum):
    """Trade action the agent can propose."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class NarrativeStage(StrEnum):
    """How mature a market narrative is."""

    EARLY = "early"
    MAINSTREAM = "mainstream"
    EXHAUSTED = "exhausted"


# === LLM Output Schemas (Pydantic models for structured output) ===


class RegimeClassification(BaseModel):
    """Output schema for the regime classification LLM call."""

    regime: Regime
    regime_confidence: float
    regime_reasoning: str
    active_narratives: list[str] = []
    narrative_stage: NarrativeStage = NarrativeStage.EARLY
    recommended_strategy: Strategy
    strategy_reasoning: str


class TradeProposal(BaseModel):
    """Output schema for the trade analysis LLM call."""

    action: Action
    asset: str
    confidence: float
    size_suggestion: str  # "small" | "medium" | "large"
    reasoning: str
    timeframe: str
    stop_loss_pct: float
    take_profit_pct: float
    key_factors: list[str]
    regime_alignment: str


class ReflectionEntry(BaseModel):
    """Output from the reflection node."""

    timestamp: datetime
    trades_reviewed: int
    regime_accuracy: str
    patterns_noticed: list[str]
    lessons_learned: list[str]


# === Market Data ===


@dataclass
class MarketSnapshot:
    """Point-in-time market data collected by the gather node."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    price: float = 0.0
    ohlcv: list[list[float]] = field(default_factory=list)  # [timestamp, O, H, L, C, V]
    volume_24h: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0

    # Technical indicators (computed locally, not by LLM)
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    bbands_upper: float | None = None
    bbands_lower: float | None = None
    volatility: float | None = None  # ATR-based
    volume_sma_ratio: float | None = None


@dataclass
class SentimentData:
    """Aggregated sentiment from various sources."""

    fear_greed_index: float | None = None  # 0-100
    news_sentiment: float | None = None  # -1 to 1
    social_sentiment: float | None = None  # -1 to 1
    news_headlines: list[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    """Current portfolio snapshot."""

    total_value: float = 0.0
    available_capital: float = 0.0
    open_positions: list[dict] = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    daily_trades: int = 0


# === LangGraph State ===


class AgentState(TypedDict, total=False):
    """The state that flows through the LangGraph workflow.

    Every node reads from and writes to this state. LangGraph
    handles persistence and checkpointing via SQLite.
    """

    # Step 1: GATHER outputs
    market: MarketSnapshot
    sentiment: SentimentData
    portfolio: PortfolioState
    recent_trades: list[dict]
    memory: list[str]  # Lessons learned from past sessions

    # Step 2: ROUTE output
    selected_model: str

    # Step 3: CLASSIFY REGIME output
    regime: RegimeClassification | None

    # Step 4: ANALYZE output
    proposal: TradeProposal | None

    # Step 5: VALIDATE output
    validation_passed: bool
    validation_reason: str

    # Step 6: EXECUTE output
    execution_result: dict | None

    # Step 7: REFLECT output
    reflection: ReflectionEntry | None

    # Metadata
    cycle_id: str
    cycle_timestamp: str
    error: str | None
