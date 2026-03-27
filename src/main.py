"""Esnaf — Regime-Aware Crypto Trading Agent.

Entry point that initializes all components and starts the agent loop.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

import structlog
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.config import load_settings
from src.data.market import MarketDataClient
from src.execution.paper_broker import PaperBroker
from src.llm.client import LLMClient
from src.risk.engine import RiskEngine
from src.storage.database import Database

logger = structlog.get_logger()
console = Console()

# Global flag for graceful shutdown
_shutdown = asyncio.Event()


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
            structlog.get_level_from_name(settings.agent.log_level)
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

    # Build the LangGraph workflow
    graph = build_graph(
        settings=settings,
        market_client=market_client,
        broker=broker,
        llm_client=llm_client,
        risk_engine=risk_engine,
        db=db,
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

            try:
                state = await run_cycle(graph, state, cycle_num)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("cycle_error", cycle=cycle_num, error=str(e))
                console.print(f"[red]Cycle {cycle_num} failed: {e}[/]")

            # Print portfolio state
            portfolio = state.get("portfolio")
            if portfolio:
                pnl_color = "green" if portfolio.daily_pnl >= 0 else "red"
                console.print(
                    f"  [dim]Portfolio:[/] ${portfolio.total_value:,.2f} "
                    f"[{pnl_color}]({portfolio.daily_pnl_pct:+.2f}%)[/] "
                    f"| Positions: {len(portfolio.open_positions)}"
                )

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
        portfolio = broker.get_portfolio_state(price)
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
