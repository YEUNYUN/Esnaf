"""SQLite database schema and operations.

All trades, decisions, portfolio snapshots, and agent memory are stored here.
Uses aiosqlite for async operations compatible with LangGraph.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import structlog

logger = structlog.get_logger()

DEFAULT_DB_PATH = Path("esnaf.db")

SCHEMA_SQL = """
-- Trades log: every executed trade
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    value REAL NOT NULL,
    fee REAL DEFAULT 0.0,
    stop_loss REAL,
    take_profit REAL,
    regime TEXT,
    strategy TEXT,
    confidence REAL,
    reasoning TEXT,
    status TEXT DEFAULT 'open'
);

-- Agent decisions: every cycle's full decision context
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    regime TEXT,
    regime_confidence REAL,
    regime_reasoning TEXT,
    action TEXT NOT NULL,
    confidence REAL,
    reasoning TEXT,
    validation_passed INTEGER NOT NULL,
    validation_reason TEXT,
    model_used TEXT,
    market_snapshot TEXT,
    indicators TEXT
);

-- Portfolio snapshots: periodic state captures
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    total_value REAL NOT NULL,
    available_capital REAL NOT NULL,
    open_positions TEXT,
    daily_pnl REAL DEFAULT 0.0,
    daily_pnl_pct REAL DEFAULT 0.0,
    total_pnl REAL DEFAULT 0.0
);

-- Agent memory: lessons learned, patterns, reflections
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT
);

-- Regime history: track regime classifications over time
CREATE TABLE IF NOT EXISTS regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    regime TEXT NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT,
    strategy TEXT,
    was_accurate INTEGER DEFAULT NULL
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_decisions_cycle ON decisions(cycle_id);
CREATE INDEX IF NOT EXISTS idx_regime_history_timestamp ON regime_history(timestamp);
"""


class Database:
    """Async SQLite database for all agent persistence."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Connect and initialize the database schema."""
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info("database_connected", path=str(self._db_path))

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def log_trade(
        self,
        cycle_id: str,
        symbol: str,
        action: str,
        side: str,
        price: float,
        quantity: float,
        value: float,
        fee: float = 0.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        regime: str | None = None,
        strategy: str | None = None,
        confidence: float | None = None,
        reasoning: str | None = None,
    ) -> int:
        """Log an executed trade."""
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO trades
            (timestamp, cycle_id, symbol, action, side, price, quantity, value,
             fee, stop_loss, take_profit, regime, strategy, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                cycle_id,
                symbol,
                action,
                side,
                price,
                quantity,
                value,
                fee,
                stop_loss,
                take_profit,
                regime,
                strategy,
                confidence,
                reasoning,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid or 0

    async def log_decision(
        self,
        cycle_id: str,
        regime: str | None,
        regime_confidence: float | None,
        regime_reasoning: str | None,
        action: str,
        confidence: float | None,
        reasoning: str | None,
        validation_passed: bool,
        validation_reason: str,
        model_used: str | None = None,
        market_snapshot: dict | None = None,
        indicators: dict | None = None,
    ) -> int:
        """Log a full decision cycle."""
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO decisions
            (timestamp, cycle_id, regime, regime_confidence, regime_reasoning,
             action, confidence, reasoning, validation_passed, validation_reason,
             model_used, market_snapshot, indicators)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                cycle_id,
                regime,
                regime_confidence,
                regime_reasoning,
                action,
                confidence,
                reasoning,
                int(validation_passed),
                validation_reason,
                model_used,
                json.dumps(market_snapshot) if market_snapshot else None,
                json.dumps(indicators) if indicators else None,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid or 0

    async def log_regime(
        self,
        cycle_id: str,
        regime: str,
        confidence: float,
        reasoning: str | None = None,
        strategy: str | None = None,
    ) -> int:
        """Log a regime classification."""
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO regime_history
            (timestamp, cycle_id, regime, confidence, reasoning, strategy)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                cycle_id,
                regime,
                confidence,
                reasoning,
                strategy,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid or 0

    async def add_memory(
        self, entry_type: str, content: str, metadata: dict | None = None
    ) -> int:
        """Add a memory entry (lesson learned, pattern, etc.)."""
        assert self._db is not None
        cursor = await self._db.execute(
            "INSERT INTO memory (timestamp, entry_type, content, metadata) VALUES (?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                entry_type,
                content,
                json.dumps(metadata) if metadata else None,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid or 0

    async def get_recent_memories(self, limit: int = 20) -> list[dict]:
        """Retrieve recent memory entries for inclusion in LLM prompts."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM memory ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_recent_trades(self, limit: int = 10) -> list[dict]:
        """Retrieve recent trades."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_open_trades(self) -> list[dict]:
        """Retrieve all open trades."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_daily_stats(self) -> dict:
        """Get today's trading statistics."""
        assert self._db is not None
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        cursor = await self._db.execute(
            """SELECT
                COUNT(*) as trade_count,
                COALESCE(SUM(CASE WHEN action IN ('BUY', 'SELL') THEN value ELSE 0 END), 0) as volume,
                COALESCE(SUM(fee), 0) as total_fees
            FROM trades WHERE timestamp LIKE ?""",
            (f"{today}%",),
        )
        row = await cursor.fetchone()
        return dict(row) if row else {"trade_count": 0, "volume": 0, "total_fees": 0}
