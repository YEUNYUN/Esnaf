"""Tests for the risk engine — the most critical component."""

from __future__ import annotations

import time

import pytest

from src.agent.state import (
    Action,
    NarrativeStage,
    PortfolioState,
    Regime,
    RegimeClassification,
    Strategy,
    TradeProposal,
)
from src.config import RiskConfig
from src.risk.engine import RiskEngine


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_position_pct=10.0,
        max_daily_drawdown_pct=3.0,
        max_total_exposure_pct=30.0,
        min_confidence_threshold=0.7,
        min_regime_confidence=0.6,
        counter_trend_confidence=0.8,
        trade_cooldown_minutes=30,
        max_trades_per_day=10,
        stop_loss_pct=3.0,
        kill_switch_drawdown_pct=5.0,
    )


@pytest.fixture
def engine(risk_config: RiskConfig) -> RiskEngine:
    return RiskEngine(config=risk_config)


@pytest.fixture
def portfolio() -> PortfolioState:
    return PortfolioState(
        total_value=10000.0,
        available_capital=8000.0,
        open_positions=[],
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        total_pnl=0.0,
        daily_trades=0,
    )


@pytest.fixture
def bull_regime() -> RegimeClassification:
    return RegimeClassification(
        regime=Regime.BULL_TREND,
        regime_confidence=0.85,
        regime_reasoning="Strong uptrend with increasing volume",
        active_narratives=["BTC ETF flows"],
        narrative_stage=NarrativeStage.MAINSTREAM,
        recommended_strategy=Strategy.TREND_FOLLOW,
        strategy_reasoning="Trend following in bull market",
    )


@pytest.fixture
def good_buy_proposal() -> TradeProposal:
    return TradeProposal(
        action=Action.BUY,
        asset="BTC/USDT",
        confidence=0.8,
        size_suggestion="small",
        reasoning="RSI oversold in bull regime, good entry",
        timeframe="4h",
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        key_factors=["RSI oversold", "bull trend"],
        regime_alignment="With trend — buying in bull market",
    )


class TestRiskEngineBasics:
    """Basic risk engine validation tests."""

    def test_valid_trade_passes(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        portfolio: PortfolioState,
        bull_regime: RegimeClassification,
    ) -> None:
        result = engine.validate(good_buy_proposal, portfolio, bull_regime)
        assert result.passed is True
        assert result.reason == "All risk checks passed"

    def test_hold_always_passes(
        self,
        engine: RiskEngine,
        portfolio: PortfolioState,
        bull_regime: RegimeClassification,
    ) -> None:
        hold = TradeProposal(
            action=Action.HOLD,
            asset="BTC/USDT",
            confidence=0.5,
            size_suggestion="small",
            reasoning="No clear signal",
            timeframe="15m",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=[],
            regime_alignment="N/A",
        )
        result = engine.validate(hold, portfolio, bull_regime)
        assert result.passed is True

    def test_low_confidence_rejected(
        self,
        engine: RiskEngine,
        portfolio: PortfolioState,
        bull_regime: RegimeClassification,
    ) -> None:
        low_conf = TradeProposal(
            action=Action.BUY,
            asset="BTC/USDT",
            confidence=0.5,
            size_suggestion="small",
            reasoning="Weak signal",
            timeframe="15m",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=[],
            regime_alignment="With trend",
        )
        result = engine.validate(low_conf, portfolio, bull_regime)
        assert result.passed is False
        assert "confidence" in result.reason.lower()


class TestRegimeChecks:
    """Tests for regime-aware risk validation."""

    def test_no_regime_rejected(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        portfolio: PortfolioState,
    ) -> None:
        result = engine.validate(good_buy_proposal, portfolio, regime=None)
        assert result.passed is False
        assert "regime" in result.reason.lower()

    def test_low_regime_confidence_rejected(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        portfolio: PortfolioState,
    ) -> None:
        weak_regime = RegimeClassification(
            regime=Regime.UNKNOWN,
            regime_confidence=0.3,
            regime_reasoning="Unclear market conditions",
            recommended_strategy=Strategy.SIT_OUT,
            strategy_reasoning="Can't classify regime",
        )
        result = engine.validate(good_buy_proposal, portfolio, weak_regime)
        assert result.passed is False
        assert "regime confidence" in result.reason.lower()

    def test_counter_trend_needs_high_confidence(
        self,
        engine: RiskEngine,
        portfolio: PortfolioState,
    ) -> None:
        bear_regime = RegimeClassification(
            regime=Regime.BEAR_TREND,
            regime_confidence=0.8,
            regime_reasoning="Clear downtrend",
            recommended_strategy=Strategy.DEFENSIVE,
            strategy_reasoning="Defensive in bear",
        )
        # BUY in bear with 0.75 confidence — should be rejected (needs 0.8+)
        counter_buy = TradeProposal(
            action=Action.BUY,
            asset="BTC/USDT",
            confidence=0.75,
            size_suggestion="small",
            reasoning="Possible reversal",
            timeframe="4h",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=["reversal pattern"],
            regime_alignment="Counter-trend",
        )
        result = engine.validate(counter_buy, portfolio, bear_regime)
        assert result.passed is False
        assert "counter-trend" in result.reason.lower()

    def test_counter_trend_passes_with_high_confidence(
        self,
        engine: RiskEngine,
        portfolio: PortfolioState,
    ) -> None:
        bear_regime = RegimeClassification(
            regime=Regime.BEAR_TREND,
            regime_confidence=0.8,
            regime_reasoning="Clear downtrend",
            recommended_strategy=Strategy.DEFENSIVE,
            strategy_reasoning="Defensive in bear",
        )
        # BUY in bear with 0.85 confidence — should pass
        strong_buy = TradeProposal(
            action=Action.BUY,
            asset="BTC/USDT",
            confidence=0.85,
            size_suggestion="small",
            reasoning="Strong reversal signal",
            timeframe="4h",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=["whale accumulation", "extreme oversold"],
            regime_alignment="Counter-trend but high conviction",
        )
        result = engine.validate(strong_buy, portfolio, bear_regime)
        assert result.passed is True


class TestDrawdownProtection:
    """Tests for drawdown and kill switch protection."""

    def test_daily_drawdown_blocks_trading(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        bull_regime: RegimeClassification,
    ) -> None:
        portfolio = PortfolioState(
            total_value=9700.0,
            available_capital=7700.0,
            daily_pnl=-300.0,
            daily_pnl_pct=-3.1,
            daily_trades=2,
        )
        result = engine.validate(good_buy_proposal, portfolio, bull_regime)
        assert result.passed is False
        assert "drawdown" in result.reason.lower()

    def test_kill_switch_triggers(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        bull_regime: RegimeClassification,
    ) -> None:
        portfolio = PortfolioState(
            total_value=9400.0,
            available_capital=7400.0,
            daily_pnl=-600.0,
            daily_pnl_pct=-5.5,
            daily_trades=3,
        )
        result = engine.validate(good_buy_proposal, portfolio, bull_regime)
        assert result.passed is False
        assert "kill switch" in result.reason.lower()
        assert engine.is_kill_switch_active

    def test_kill_switch_blocks_all_future_trades(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        portfolio: PortfolioState,
        bull_regime: RegimeClassification,
    ) -> None:
        # Trigger kill switch
        engine._kill_switch_active = True

        # Even a perfectly valid trade should be blocked
        result = engine.validate(good_buy_proposal, portfolio, bull_regime)
        assert result.passed is False
        assert "kill switch active" in result.reason.lower()

    def test_kill_switch_manual_reset(self, engine: RiskEngine) -> None:
        engine._kill_switch_active = True
        assert engine.is_kill_switch_active
        engine.reset_kill_switch()
        assert not engine.is_kill_switch_active


class TestPositionLimits:
    """Tests for position size and exposure limits."""

    def test_max_daily_trades(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        bull_regime: RegimeClassification,
    ) -> None:
        portfolio = PortfolioState(
            total_value=10000.0,
            available_capital=8000.0,
            daily_trades=10,  # Already hit the limit
        )
        result = engine.validate(good_buy_proposal, portfolio, bull_regime)
        assert result.passed is False
        assert "daily trade limit" in result.reason.lower()

    def test_total_exposure_limit(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        bull_regime: RegimeClassification,
    ) -> None:
        # Portfolio already at 28% exposure (near the 30% limit)
        portfolio = PortfolioState(
            total_value=10000.0,
            available_capital=5000.0,
            open_positions=[
                {"value": 1400},
                {"value": 1400},
            ],
            daily_trades=1,
        )
        # "small" = 3% of 10000 = 300. Current exposure = 2800. Total = 3100 > 3000 limit
        result = engine.validate(good_buy_proposal, portfolio, bull_regime)
        assert result.passed is False
        assert "exposure" in result.reason.lower()


class TestCooldown:
    """Tests for trade cooldown."""

    def test_cooldown_blocks_rapid_trades(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        portfolio: PortfolioState,
        bull_regime: RegimeClassification,
    ) -> None:
        # Record a recent trade
        engine.record_trade()

        result = engine.validate(good_buy_proposal, portfolio, bull_regime)
        assert result.passed is False
        assert "cooldown" in result.reason.lower()

    def test_no_cooldown_after_waiting(
        self,
        engine: RiskEngine,
        good_buy_proposal: TradeProposal,
        portfolio: PortfolioState,
        bull_regime: RegimeClassification,
    ) -> None:
        # Record a trade from 31+ minutes ago
        engine._recent_trade_timestamps = [time.time() - 1860]

        result = engine.validate(good_buy_proposal, portfolio, bull_regime)
        assert result.passed is True
