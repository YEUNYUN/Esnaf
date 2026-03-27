"""Esnaf configuration — all settings loaded from environment variables.

Uses pydantic-settings for validation. Every setting has a sensible default
for paper trading. Only exchange API keys are required for live mode.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeConfig(BaseSettings):
    """Exchange API configuration."""

    model_config = SettingsConfigDict(env_prefix="BINANCE_")

    api_key: str = ""
    secret: str = ""
    testnet: bool = True


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    copilot_api_base: str = Field(
        default="http://localhost:4141/v1", alias="COPILOT_API_BASE"
    )

    # Model identifiers for LiteLLM
    primary_model: str = "openai/claude-sonnet-4-20250514"  # Via Copilot proxy
    cheap_model: str = "gemini/gemini-2.5-flash"
    fallback_model: str = "ollama/llama3.1"

    # Routing thresholds
    calm_volatility_threshold: float = 0.02  # Below this → use cheap model
    temperature: float = 0.0


class RiskConfig(BaseSettings):
    """Risk management hard limits. These CANNOT be overridden by the LLM."""

    model_config = SettingsConfigDict(env_prefix="", populate_by_name=True)

    max_capital: float = Field(default=10000.0, alias="MAX_CAPITAL")
    max_position_pct: float = Field(default=10.0, alias="MAX_POSITION_PCT")
    max_daily_drawdown_pct: float = Field(default=3.0, alias="MAX_DAILY_DRAWDOWN_PCT")
    max_total_exposure_pct: float = 30.0
    min_confidence_threshold: float = Field(default=0.7, alias="MIN_CONFIDENCE_THRESHOLD")
    min_regime_confidence: float = 0.6
    counter_trend_confidence: float = 0.8
    trade_cooldown_minutes: int = 30
    max_trades_per_day: int = 10
    stop_loss_pct: float = 3.0
    max_take_profit_pct: float = 50.0
    max_directional_exposure_pct: float = 20.0
    kill_switch_drawdown_pct: float = 5.0
    fee_rate: float = 0.001  # 0.1% per trade (Binance default)


class AgentConfig(BaseSettings):
    """Agent behavior configuration."""

    model_config = SettingsConfigDict(env_prefix="", populate_by_name=True)

    trading_pair: str = Field(default="BTC/USDT", alias="TRADING_PAIR")
    cycle_interval_minutes: int = Field(default=15, alias="CYCLE_INTERVAL")
    analysis_interval_minutes: int = Field(default=15, alias="ANALYSIS_INTERVAL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Paper trading is the default — must explicitly enable live
    paper_trading: bool = True
    initial_capital: float = 10000.0  # Paper trading starting capital


class Settings(BaseSettings):
    """Root settings — aggregates all config sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_settings() -> Settings:
    """Load and validate all settings from environment."""
    return Settings()
