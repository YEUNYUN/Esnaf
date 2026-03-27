"""Tests for the agent state models."""

from __future__ import annotations

from src.agent.state import (
    Action,
    MarketSnapshot,
    NarrativeStage,
    Regime,
    RegimeClassification,
    Strategy,
    TradeProposal,
)


class TestEnums:
    def test_regime_values(self) -> None:
        assert Regime.BULL_TREND.value == "bull_trend"
        assert Regime.BEAR_TREND.value == "bear_trend"
        assert Regime.RANGING.value == "ranging"

    def test_strategy_values(self) -> None:
        assert Strategy.SIT_OUT.value == "sit_out"
        assert Strategy.TREND_FOLLOW.value == "trend_follow"

    def test_action_values(self) -> None:
        assert Action.BUY.value == "BUY"
        assert Action.HOLD.value == "HOLD"


class TestRegimeClassification:
    def test_create_classification(self) -> None:
        rc = RegimeClassification(
            regime=Regime.BULL_TREND,
            regime_confidence=0.85,
            regime_reasoning="Clear uptrend",
            recommended_strategy=Strategy.TREND_FOLLOW,
            strategy_reasoning="Following the trend",
        )
        assert rc.regime == Regime.BULL_TREND
        assert rc.regime_confidence == 0.85
        assert rc.narrative_stage == NarrativeStage.EARLY  # default

    def test_classification_with_narratives(self) -> None:
        rc = RegimeClassification(
            regime=Regime.BULL_TREND,
            regime_confidence=0.9,
            regime_reasoning="AI token narrative driving market",
            active_narratives=["AI tokens", "L2 season"],
            narrative_stage=NarrativeStage.MAINSTREAM,
            recommended_strategy=Strategy.TREND_FOLLOW,
            strategy_reasoning="Ride the narrative",
        )
        assert len(rc.active_narratives) == 2
        assert rc.narrative_stage == NarrativeStage.MAINSTREAM


class TestTradeProposal:
    def test_create_proposal(self) -> None:
        tp = TradeProposal(
            action=Action.BUY,
            asset="BTC/USDT",
            confidence=0.8,
            size_suggestion="medium",
            reasoning="Strong signal",
            timeframe="4h",
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            key_factors=["RSI", "volume"],
            regime_alignment="With trend",
        )
        assert tp.action == Action.BUY
        assert tp.confidence == 0.8
        assert len(tp.key_factors) == 2


class TestMarketSnapshot:
    def test_default_snapshot(self) -> None:
        ms = MarketSnapshot()
        assert ms.price == 0.0
        assert ms.rsi is None
        assert ms.volatility is None

    def test_snapshot_with_indicators(self) -> None:
        ms = MarketSnapshot(
            symbol="BTC/USDT",
            price=50000.0,
            rsi=35.5,
            macd=-100.0,
            volatility=0.025,
        )
        assert ms.rsi == 35.5
        assert ms.volatility == 0.025
