"""Tests for configuration loading."""

from __future__ import annotations

from src.config import (
    AgentConfig,
    ExchangeConfig,
    LLMConfig,
    RiskConfig,
    load_settings,
)


class TestConfigDefaults:
    """Test that all configs have sensible defaults for paper trading."""

    def test_exchange_defaults_to_testnet(self) -> None:
        config = ExchangeConfig()
        assert config.testnet is True
        assert config.api_key == ""

    def test_risk_defaults(self) -> None:
        config = RiskConfig()
        assert config.max_position_pct == 10.0
        assert config.max_daily_drawdown_pct == 3.0
        assert config.min_confidence_threshold == 0.7
        assert config.min_regime_confidence == 0.6
        assert config.kill_switch_drawdown_pct == 5.0

    def test_agent_defaults_to_paper_trading(self) -> None:
        config = AgentConfig()
        assert config.paper_trading is True
        assert config.trading_pair == "BTC/USDT"
        assert config.analysis_interval_minutes == 15

    def test_llm_defaults(self) -> None:
        config = LLMConfig()
        assert "claude" in config.primary_model
        assert "gemini" in config.cheap_model
        assert "ollama" in config.fallback_model
        assert config.temperature == 0.0

    def test_settings_loads(self) -> None:
        settings = load_settings()
        assert settings.agent.paper_trading is True
        assert settings.exchange.testnet is True
        assert settings.risk.max_position_pct > 0


class TestConfigValidation:
    def test_risk_limits_are_reasonable(self) -> None:
        config = RiskConfig()
        assert config.kill_switch_drawdown_pct > config.max_daily_drawdown_pct
        assert config.counter_trend_confidence > config.min_confidence_threshold
        assert config.min_regime_confidence > 0.0
        assert config.max_total_exposure_pct > config.max_position_pct
