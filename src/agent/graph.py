"""LangGraph workflow — the complete agent reasoning loop.

This is the central orchestrator. It wires all nodes together into
the regime-first architecture:

  GATHER → ROUTE → CLASSIFY REGIME → ANALYZE → VALIDATE → EXECUTE → REFLECT
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import structlog

from src.agent.nodes.analyze import analyze_node
from src.agent.nodes.classify import classify_node
from src.agent.nodes.execute import execute_node
from src.agent.nodes.gather import gather_node
from src.agent.nodes.reflect import reflect_node
from src.agent.nodes.validate import validate_node
from src.agent.router import select_model
from src.agent.state import AgentState, Strategy
from src.config import Settings
from src.data.market import MarketDataClient
from src.data.sentiment import SentimentPipeline
from src.execution.paper_broker import PaperBroker
from src.llm.client import LLMClient
from src.risk.engine import RiskEngine
from src.storage.database import Database

logger = structlog.get_logger()


def route_model_node(state: AgentState, *, settings: Settings) -> dict:
    """Deterministic model selection — no LLM involvement."""
    market = state.get("market")
    portfolio = state.get("portfolio")
    model = select_model(settings.llm, market, portfolio)
    return {"selected_model": model}


def should_analyze(state: AgentState) -> Literal["analyze", "skip_to_execute"]:
    """Conditional edge: skip analysis if regime says SIT_OUT."""
    regime = state.get("regime")
    if regime and regime.recommended_strategy != Strategy.SIT_OUT:
        return "analyze"
    return "skip_to_execute"


def should_execute(state: AgentState) -> Literal["execute", "skip_to_reflect"]:
    """Conditional edge: skip execution if validation failed."""
    if state.get("validation_passed", False):
        return "execute"
    return "skip_to_reflect"


def _make_classify_node(llm_client: LLMClient, settings: Settings):
    """Create an async classify node with injected dependencies."""
    async def _classify(state: AgentState) -> dict:
        model = state.get("selected_model", settings.llm.primary_model)
        return await classify_node(state, llm_client=llm_client, model=model)
    return _classify


def _make_analyze_node(llm_client: LLMClient, settings: Settings):
    """Create an async analyze node with injected dependencies."""
    async def _analyze(state: AgentState) -> dict:
        model = state.get("selected_model", settings.llm.primary_model)
        return await analyze_node(state, llm_client=llm_client, model=model)
    return _analyze


def _make_reflect_node(llm_client: LLMClient, settings: Settings, db: Database):
    """Create an async reflect node with injected dependencies."""
    async def _reflect(state: AgentState) -> dict:
        return await reflect_node(
            state, llm_client=llm_client, model=settings.llm.cheap_model, db=db,
        )
    return _reflect


def build_graph(
    settings: Settings,
    market_client: MarketDataClient,
    broker: PaperBroker,
    llm_client: LLMClient,
    risk_engine: RiskEngine,
    db: Database,
    sentiment_pipeline: SentimentPipeline | None = None,
):
    """Build and compile the LangGraph workflow.

    Returns a compiled graph that can be invoked with an empty state.
    """
    from langgraph.graph import END, StateGraph

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes with their dependencies injected via partial
    workflow.add_node(
        "gather",
        partial(
            gather_node,
            market_client=market_client,
            broker=broker,
            sentiment_pipeline=sentiment_pipeline,
            symbol=settings.agent.trading_pair,
        ),
    )

    workflow.add_node(
        "route",
        partial(route_model_node, settings=settings),
    )

    workflow.add_node(
        "classify",
        _make_classify_node(llm_client, settings),
    )

    workflow.add_node(
        "analyze",
        _make_analyze_node(llm_client, settings),
    )

    workflow.add_node(
        "validate",
        partial(validate_node, risk_engine=risk_engine),
    )

    workflow.add_node(
        "execute",
        partial(execute_node, broker=broker, risk_engine=risk_engine, db=db),
    )

    workflow.add_node(
        "reflect",
        _make_reflect_node(llm_client, settings, db),
    )

    # Define edges: the regime-first flow
    workflow.set_entry_point("gather")
    workflow.add_edge("gather", "route")
    workflow.add_edge("route", "classify")

    # After classify: either analyze (if strategy != SIT_OUT) or skip to execute
    workflow.add_conditional_edges("classify", should_analyze, {
        "analyze": "analyze",
        "skip_to_execute": "execute",
    })

    workflow.add_edge("analyze", "validate")

    # After validate: either execute (if passed) or skip to reflect
    workflow.add_conditional_edges("validate", should_execute, {
        "execute": "execute",
        "skip_to_reflect": "reflect",
    })

    workflow.add_edge("execute", "reflect")
    workflow.add_edge("reflect", END)

    # Compile
    compiled = workflow.compile()
    logger.info("graph_compiled", nodes=7, edges=8)
    return compiled
