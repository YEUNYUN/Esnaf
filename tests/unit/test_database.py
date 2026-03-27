"""Tests for Database helper methods added in Tasks 1-3."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.database import Database


@pytest.fixture
async def db(tmp_path: Path):
    """Create an in-memory-like temporary database."""
    d = Database(db_path=tmp_path / "test.db")
    await d.connect()
    yield d
    await d.close()


class TestGetRecentMemories:
    """Task 1 — entry_type filtering."""

    @pytest.mark.asyncio
    async def test_no_filter_returns_all(self, db: Database):
        await db.add_memory("lesson", "lesson 1")
        await db.add_memory("pattern", "pattern 1")

        rows = await db.get_recent_memories(limit=10)
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_filter_by_entry_type(self, db: Database):
        await db.add_memory("lesson", "lesson 1")
        await db.add_memory("pattern", "pattern 1")
        await db.add_memory("lesson", "lesson 2")

        lessons = await db.get_recent_memories(limit=10, entry_type="lesson")
        assert len(lessons) == 2
        assert all(r["entry_type"] == "lesson" for r in lessons)

    @pytest.mark.asyncio
    async def test_filter_returns_empty_for_missing_type(self, db: Database):
        await db.add_memory("lesson", "lesson 1")

        rows = await db.get_recent_memories(limit=10, entry_type="nonexistent")
        assert rows == []


class TestGetRegimeHistorySummary:
    """Task 1 — regime history summary via public API."""

    @pytest.mark.asyncio
    async def test_no_data(self, db: Database):
        summary = await db.get_regime_history_summary()
        assert summary == "No prior regime classifications."

    @pytest.mark.asyncio
    async def test_summary_counts(self, db: Database):
        await db.log_regime("c1", "bull_trend", 0.8)
        await db.log_regime("c2", "bull_trend", 0.9)
        await db.log_regime("c3", "bear_trend", 0.7)

        summary = await db.get_regime_history_summary(limit=10)
        assert "Last 3 classifications:" in summary
        assert "bull_trend: 2/3" in summary
        assert "bear_trend: 1/3" in summary


class TestSavePortfolioSnapshot:
    """Task 3 — portfolio snapshot persistence."""

    @pytest.mark.asyncio
    async def test_save_and_verify(self, db: Database):
        await db.save_portfolio_snapshot(
            total_value=10500.0,
            available_capital=8000.0,
            open_positions=2,
            unrealized_pnl=500.0,
            cycle_id=1,
        )
        assert db._db is not None
        cursor = await db._db.execute("SELECT * FROM portfolio_snapshots")
        rows = await cursor.fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["total_value"] == 10500.0
        assert row["available_capital"] == 8000.0
        assert row["open_positions"] == "2"

    @pytest.mark.asyncio
    async def test_multiple_snapshots(self, db: Database):
        for i in range(3):
            await db.save_portfolio_snapshot(
                total_value=10000.0 + i * 100,
                available_capital=8000.0,
                open_positions=i,
                unrealized_pnl=float(i * 50),
            )
        assert db._db is not None
        cursor = await db._db.execute("SELECT COUNT(*) as cnt FROM portfolio_snapshots")
        row = await cursor.fetchone()
        assert dict(row)["cnt"] == 3


class TestAgentMemoryRegimeSummary:
    """Task 2 — AgentMemory delegates to Database."""

    @pytest.mark.asyncio
    async def test_delegates_to_db(self, db: Database):
        from src.agent.memory import AgentMemory

        await db.log_regime("c1", "ranging", 0.6)
        memory = AgentMemory(db)
        summary = await memory.get_regime_history_summary(limit=5)
        assert "ranging" in summary
