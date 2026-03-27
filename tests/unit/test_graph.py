"""Tests for the LangGraph agent workflow."""

from __future__ import annotations

from src.agent.graph import should_analyze, should_execute
from src.agent.state import (
    AgentState,
    Regime,
    RegimeClassification,
    Strategy,
)


class TestGraphRouting:
    """Test conditional edge functions."""

    def test_should_analyze_when_not_sit_out(self):
        state: AgentState = {
            "regime": RegimeClassification(
                regime=Regime.BULL_TREND,
                regime_confidence=0.8,
                regime_reasoning="Strong uptrend",
                recommended_strategy=Strategy.TREND_FOLLOW,
                strategy_reasoning="Follow the trend",
            )
        }
        assert should_analyze(state) == "analyze"

    def test_should_skip_analyze_when_sit_out(self):
        state: AgentState = {
            "regime": RegimeClassification(
                regime=Regime.UNKNOWN,
                regime_confidence=0.3,
                regime_reasoning="Unclear regime",
                recommended_strategy=Strategy.SIT_OUT,
                strategy_reasoning="Too uncertain",
            )
        }
        assert should_analyze(state) == "skip_to_execute"

    def test_should_skip_analyze_when_no_regime(self):
        state: AgentState = {"regime": None}
        assert should_analyze(state) == "skip_to_execute"

    def test_should_execute_when_validation_passed(self):
        state: AgentState = {"validation_passed": True}
        assert should_execute(state) == "execute"

    def test_should_skip_execute_when_validation_failed(self):
        state: AgentState = {"validation_passed": False}
        assert should_execute(state) == "skip_to_reflect"

    def test_should_skip_execute_when_no_validation(self):
        state: AgentState = {}
        assert should_execute(state) == "skip_to_reflect"


class TestGraphBuild:
    """Test that the graph compiles correctly."""

    def test_graph_compiles(self):
        """Verify the graph can be built with mock dependencies."""
        from unittest.mock import MagicMock

        from src.agent.graph import build_graph
        from src.config import Settings

        settings = Settings()
        graph = build_graph(
            settings=settings,
            market_client=MagicMock(),
            broker=MagicMock(),
            llm_client=MagicMock(),
            risk_engine=MagicMock(),
            db=MagicMock(),
        )

        assert graph is not None
        # LangGraph compiled graph should have invoke/ainvoke
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")
