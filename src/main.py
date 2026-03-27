"""Esnaf — Regime-Aware Crypto Trading Agent.

Entry point that initializes all components and starts the agent loop.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import logging
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
import structlog

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.config import load_settings
from src.data.market import MarketDataClient
from src.data.sentiment import SentimentPipeline
from src.execution.paper_broker import PaperBroker
from src.llm.client import LLMClient
from src.risk.engine import RiskEngine
from src.storage.database import Database

logger = structlog.get_logger()
console = Console()

# Global flag for graceful shutdown
_shutdown = asyncio.Event()

# ── Cycle error-recovery constants ──────────────────────────────────────────
_CONSECUTIVE_FAILURE_PAUSE_THRESHOLD = 3
_CONSECUTIVE_FAILURE_PAUSE_SECONDS = 300  # 5 minutes
_MAX_CONSECUTIVE_FAILURES = 10
_TRANSIENT_RETRY_DELAY_SECONDS = 5

# Fields that belong to a single cycle and must not leak into the next one
_CYCLE_FIELDS = frozenset({
    "regime", "proposal", "validation_passed", "validation_reason",
    "execution_result", "reflection", "cycle_id", "cycle_timestamp",
    "error", "selected_model", "market", "sentiment",
})


def _classify_error(exc: Exception) -> str:
    """Classify an exception as ``transient``, ``fatal``, or ``data``.

    * **transient** — network hiccups, timeouts, rate-limits → retry once
    * **fatal**     — auth / config errors → graceful shutdown
    * **data**      — parse or stale-data errors → skip cycle, continue
    """
    exc_module = getattr(type(exc), "__module__", "") or ""
    exc_name = type(exc).__name__
    msg = str(exc).lower()

    # Transient: network, timeout, rate-limit
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return "transient"
    if "ccxt" in exc_module and exc_name in {
        "NetworkError", "RequestTimeout", "ExchangeNotAvailable",
        "DDoSProtection", "RateLimitExceeded",
    }:
        return "transient"
    if "httpx" in exc_module and ("Timeout" in exc_name or "Connect" in exc_name):
        return "transient"
    if any(kw in msg for kw in ("timeout", "rate limit", "rate_limit", "connection reset")):
        return "transient"

    # Fatal: auth, config, permission
    if "ccxt" in exc_module and exc_name in {"AuthenticationError", "PermissionDenied"}:
        return "fatal"
    if any(kw in msg for kw in ("authentication", "invalid api key", "permission denied")):
        return "fatal"
    if isinstance(exc, PermissionError):
        return "fatal"

    # Data: parse, stale, missing fields
    if isinstance(exc, (json.JSONDecodeError, KeyError, IndexError, ValueError)):
        return "data"
    if any(kw in msg for kw in ("parse", "decode", "stale", "invalid data")):
        return "data"

    # Unknown → treat as data (skip cycle, keep running)
    return "data"


def _clean_cycle_state(state: AgentState) -> AgentState:
    """Strip cycle-specific fields so partial state never carries over."""
    return {k: v for k, v in state.items() if k not in _CYCLE_FIELDS}


async def run_cycle(graph, state: AgentState, cycle_num: int) -> AgentState:
    """Run a single trading cycle through the graph."""
    console.print(f"\n[bold]── Cycle {cycle_num} ──[/] {datetime.now(UTC).strftime('%H:%M:%S UTC')}")

    result = await graph.ainvoke(state)

    # Print cycle summary
    regime = result.get("regime")
    proposal = result.get("proposal")
    validation = result.get("validation_passed", False)
    execution = result.get("execution_result", {})

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    if regime:
        confidence_color = "green" if regime.regime_confidence >= 0.7 else "yellow" if regime.regime_confidence >= 0.5 else "red"
        table.add_row("Regime", f"[bold]{regime.regime.value}[/] [{confidence_color}]({regime.regime_confidence:.0%})[/]")
        table.add_row("Strategy", regime.recommended_strategy.value)

    if proposal:
        action_color = {"BUY": "green", "SELL": "red", "HOLD": "dim", "CLOSE_LONG": "yellow", "CLOSE_SHORT": "yellow"}.get(proposal.action.value, "white")
        table.add_row("Action", f"[{action_color}]{proposal.action.value}[/] (conf: {proposal.confidence:.0%})")
        table.add_row("Reasoning", proposal.reasoning[:120])

    table.add_row("Validation", "[green]PASSED[/]" if validation else f"[red]REJECTED[/] {result.get('validation_reason', '')[:80]}")

    if execution and execution.get("status") == "filled":
        table.add_row("Executed", f"${execution.get('value', 0):,.2f} at ${execution.get('price', 0):,.2f}")

    console.print(table)
    return result


async def run() -> None:
    """Start the Esnaf agent."""
    load_dotenv()
    settings = load_settings()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.agent.log_level)
        ),
    )

    console.print("[bold green]╔══════════════════════════════════════╗[/]")
    console.print("[bold green]║  ESNAF — Crypto Trading Agent       ║[/]")
    console.print("[bold green]║  Regime-Aware • LLM-Powered         ║[/]")
    console.print("[bold green]╚══════════════════════════════════════╝[/]")
    console.print()

    console.print(f"  Trading pair: [cyan]{settings.agent.trading_pair}[/]")
    console.print(f"  Cycle interval: [cyan]{settings.agent.cycle_interval_minutes}min[/]")
    console.print(f"  Mode: [yellow]{'Paper Trading' if settings.agent.paper_trading else 'LIVE'}[/]")
    console.print(f"  Primary model: [cyan]{settings.llm.primary_model}[/]")
    console.print(f"  Capital: [cyan]${settings.risk.max_capital:,.2f}[/]")
    console.print()

    # Initialize components
    console.print("[dim]Initializing components...[/]")

    db = Database()
    await db.connect()

    market_client = MarketDataClient(config=settings.exchange)

    broker = PaperBroker(
        initial_capital=settings.risk.max_capital,
        fee_rate=settings.risk.fee_rate,
    )

    llm_client = LLMClient(settings.llm)
    risk_engine = RiskEngine(settings.risk)
    sentiment = SentimentPipeline()

    # Build the LangGraph workflow
    graph = build_graph(
        settings=settings,
        market_client=market_client,
        broker=broker,
        llm_client=llm_client,
        risk_engine=risk_engine,
        db=db,
        sentiment_pipeline=sentiment,
    )

    console.print("[green]All components initialized ✓[/]")

    logger.info(
        "esnaf_starting",
        pair=settings.agent.trading_pair,
        paper_trading=settings.agent.paper_trading,
        interval=f"{settings.agent.cycle_interval_minutes}min",
    )

    # Main loop
    cycle_num = 0
    consecutive_failures = 0
    current_date = datetime.now(UTC).date()
    state: AgentState = {}

    # Load memory from DB for continuity across restarts
    memories = await db.get_recent_memories(limit=20)
    if memories:
        state["memory"] = [m["content"] for m in memories]
        console.print(f"[dim]Loaded {len(memories)} memories from previous sessions[/]")

    recent_trades = await db.get_recent_trades(limit=5)
    if recent_trades:
        state["recent_trades"] = recent_trades

    console.print("[dim]Starting trading loop (Ctrl+C to stop)...[/]\n")

    try:
        while not _shutdown.is_set():
            cycle_num += 1

            # ── Daily stats reset at midnight UTC ────────────────────
            today = datetime.now(UTC).date()
            if today != current_date:
                try:
                    price = await market_client.fetch_price(
                        settings.agent.trading_pair,
                    )
                    await broker.reset_daily_stats(price)
                    logger.info(
                        "daily_stats_reset",
                        previous_date=str(current_date),
                        new_date=str(today),
                        reset_price=price,
                    )
                    console.print(
                        f"  [cyan]Daily stats reset for {today} "
                        f"(price ${price:,.2f})[/]"
                    )
                except Exception as e:
                    logger.error("daily_reset_failed", error=str(e))
                current_date = today

            # ── Consecutive-failure back-off ─────────────────────────
            if consecutive_failures >= _CONSECUTIVE_FAILURE_PAUSE_THRESHOLD:
                pause_min = _CONSECUTIVE_FAILURE_PAUSE_SECONDS // 60
                logger.warning(
                    "consecutive_failure_pause",
                    failures=consecutive_failures,
                    pause_minutes=pause_min,
                )
                console.print(
                    f"  [yellow]{consecutive_failures} consecutive failures "
                    f"— pausing {pause_min}min before retry[/]"
                )
                try:
                    await asyncio.wait_for(
                        _shutdown.wait(),
                        timeout=_CONSECUTIVE_FAILURE_PAUSE_SECONDS,
                    )
                    break  # shutdown requested during pause
                except TimeoutError:
                    pass  # pause finished, continue

            # ── Clean state: drop cycle-specific fields ──────────────
            state = _clean_cycle_state(state)
            pre_cycle_state = dict(state)

            try:
                state = await run_cycle(graph, state, cycle_num)
                consecutive_failures = 0

            except KeyboardInterrupt:
                break

            except Exception as e:
                error_kind = _classify_error(e)

                if error_kind == "transient":
                    logger.warning(
                        "cycle_transient_error",
                        cycle=cycle_num,
                        error=str(e),
                    )
                    console.print(
                        f"  [yellow]Cycle {cycle_num} transient error: {e} "
                        "— retrying…[/]"
                    )
                    await asyncio.sleep(_TRANSIENT_RETRY_DELAY_SECONDS)
                    try:
                        state = await run_cycle(
                            graph, pre_cycle_state, cycle_num,
                        )
                        consecutive_failures = 0
                    except Exception as retry_exc:
                        logger.warning(
                            "cycle_retry_failed",
                            cycle=cycle_num,
                            error=str(retry_exc),
                        )
                        state = pre_cycle_state
                        consecutive_failures += 1

                elif error_kind == "fatal":
                    logger.error(
                        "cycle_fatal_error",
                        cycle=cycle_num,
                        error=str(e),
                    )
                    console.print(
                        f"  [bold red]Fatal error in cycle {cycle_num}: {e}[/]"
                    )
                    _shutdown.set()
                    break

                else:  # data error
                    logger.warning(
                        "cycle_data_error",
                        cycle=cycle_num,
                        error=str(e),
                    )
                    console.print(
                        f"  [yellow]Cycle {cycle_num} data error (skipped): "
                        f"{e}[/]"
                    )
                    state = pre_cycle_state
                    consecutive_failures += 1

                # Check max consecutive failures
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "max_consecutive_failures_reached",
                        failures=consecutive_failures,
                    )
                    console.print(
                        f"  [bold red]{consecutive_failures} consecutive "
                        "failures — shutting down[/]"
                    )
                    _shutdown.set()
                    break

            # Print portfolio state and save snapshot
            portfolio = state.get("portfolio")
            if portfolio:
                pnl_color = "green" if portfolio.daily_pnl >= 0 else "red"
                console.print(
                    f"  [dim]Portfolio:[/] ${portfolio.total_value:,.2f} "
                    f"[{pnl_color}]({portfolio.daily_pnl_pct:+.2f}%)[/] "
                    f"| Positions: {len(portfolio.open_positions)}"
                )
                try:
                    unrealized = sum(
                        p.get("unrealized_pnl", 0.0) for p in portfolio.open_positions
                    )
                    await db.save_portfolio_snapshot(
                        total_value=portfolio.total_value,
                        available_capital=portfolio.available_capital,
                        open_positions=len(portfolio.open_positions),
                        unrealized_pnl=unrealized,
                        cycle_id=cycle_num,
                    )
                except Exception as snap_err:
                    logger.warning("snapshot_error", error=str(snap_err))

            # Wait for next cycle
            wait_seconds = settings.agent.cycle_interval_minutes * 60
            console.print(f"  [dim]Next cycle in {settings.agent.cycle_interval_minutes}min...[/]")

            try:
                await asyncio.wait_for(_shutdown.wait(), timeout=wait_seconds)
                break
            except TimeoutError:
                pass  # Normal — time for next cycle

    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[yellow]Shutting down...[/]")

        market = state.get("market")
        price = market.price if market else 0
        portfolio = await broker.get_portfolio_state(price)
        console.print(f"  Final portfolio value: [bold]${portfolio.total_value:,.2f}[/]")
        console.print(f"  Total P&L: [{'green' if portfolio.total_pnl >= 0 else 'red'}]{portfolio.total_pnl:+.2f}[/]")
        console.print(f"  Cycles completed: {cycle_num}")

        await db.close()
        console.print("[green]Goodbye! 👋[/]")


def main() -> None:
    """Sync entry point."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("esnaf_shutdown", reason="keyboard_interrupt")
        sys.exit(0)


if __name__ == "__main__":
    main()
