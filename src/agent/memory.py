"""Agent memory system — persistent lessons and pattern recognition.

The agent stores lessons learned from each trading cycle and loads
them into future prompts. This is how the agent "learns" over time
without fine-tuning — it's prompt-based learning via retrieved memory.
"""

from __future__ import annotations

import structlog

from src.storage.database import Database

logger = structlog.get_logger()


class AgentMemory:
    """Manages the agent's persistent memory for learning across cycles."""

    def __init__(self, db: Database, max_context_entries: int = 15) -> None:
        self._db = db
        self._max_context = max_context_entries

    async def get_context(self) -> list[str]:
        """Retrieve recent memories formatted for inclusion in LLM prompts.

        Returns the most recent lessons as a flat list of strings.
        Limited to max_context_entries to manage prompt size.
        """
        memories = await self._db.get_recent_memories(limit=self._max_context)

        # Format as simple strings for prompt inclusion
        entries = []
        for mem in memories:
            content = mem.get("content", "")
            entry_type = mem.get("entry_type", "lesson")
            if content:
                entries.append(f"[{entry_type}] {content}")

        logger.debug("memory_loaded", count=len(entries))
        return entries

    async def add_lesson(self, lesson: str, metadata: dict | None = None) -> None:
        """Store a new lesson learned from trading experience."""
        await self._db.add_memory(
            entry_type="lesson",
            content=lesson,
            metadata=metadata,
        )
        logger.info("lesson_stored", content=lesson[:80])

    async def add_pattern(self, pattern: str, metadata: dict | None = None) -> None:
        """Store a recognized pattern."""
        await self._db.add_memory(
            entry_type="pattern",
            content=pattern,
            metadata=metadata,
        )
        logger.info("pattern_stored", content=pattern[:80])

    async def get_regime_history_summary(self, limit: int = 20) -> str:
        """Get a summary of recent regime classifications for context."""
        return await self._db.get_regime_history_summary(limit=limit)
