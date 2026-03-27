"""Main entry point for Esnaf trading agent."""

from __future__ import annotations

import asyncio
import sys

import structlog
from dotenv import load_dotenv

from src.config import load_settings

logger = structlog.get_logger()


async def run() -> None:
    """Main async entry point."""
    load_dotenv()
    settings = load_settings()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(settings.agent.log_level)
        ),
    )

    logger.info(
        "esnaf_starting",
        pair=settings.agent.trading_pair,
        paper_trading=settings.agent.paper_trading,
        interval=f"{settings.agent.analysis_interval_minutes}min",
    )

    if settings.agent.paper_trading:
        logger.info("mode_paper", capital=settings.agent.initial_capital)
    else:
        logger.warning("mode_live", exchange="binance", testnet=settings.exchange.testnet)

    # TODO: Initialize components and start the agent loop
    # This will be wired up in the langgraph-agent todo
    logger.info("esnaf_ready", message="Agent initialized. Waiting for LangGraph wiring.")


def main() -> None:
    """Sync entry point."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("esnaf_shutdown", reason="keyboard_interrupt")
        sys.exit(0)


if __name__ == "__main__":
    main()
