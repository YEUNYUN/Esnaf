"""Narrative detection pipeline — detect emerging crypto narratives.

Crypto is narrative-driven. "AI coins," "RWA tokens," "L2 season," "DePIN" —
these stories drive 10-100x moves before fundamentals catch up.

This module scans free sources (CryptoPanic, Reddit) and uses the LLM to
classify narrative lifecycle stages (early → mainstream → exhausted).

An LLM's edge: it can distinguish "everyone talking about ETH" (bullish, new interest)
from "everyone talking about ETH" (bearish, euphoria top signal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from src.llm.client import LLMClient

logger = structlog.get_logger()

# Known narrative categories to track
NARRATIVE_CATEGORIES = [
    "AI/ML tokens",
    "L2/scaling",
    "DePIN",
    "RWA (Real World Assets)",
    "Memecoins",
    "BTC ETF flows",
    "Stablecoin regulation",
    "DeFi revival",
    "Gaming/Metaverse",
    "Restaking/EigenLayer",
    "Solana ecosystem",
    "Bitcoin L2/Ordinals",
    "Privacy coins",
    "Cross-chain/interop",
]

NARRATIVE_DETECTION_PROMPT = """Analyze these recent crypto headlines and social posts to detect active market narratives.

## Headlines & Posts
{headlines}

## Known Narrative Categories
{categories}

Identify which narratives are currently active and at what stage.

Respond with ONLY a JSON object:
{{
  "narratives": [
    {{
      "name": "narrative name",
      "stage": "early | mainstream | exhausted",
      "confidence": 0.0 to 1.0,
      "evidence": "brief evidence from the headlines",
      "related_tokens": ["TOKEN1", "TOKEN2"],
      "sentiment": "bullish | bearish | neutral",
      "momentum": "accelerating | steady | decelerating"
    }}
  ],
  "dominant_narrative": "the single strongest narrative right now",
  "contrarian_signal": "any narrative that seems overheated or about to reverse"
}}"""


@dataclass
class NarrativeSignal:
    """A detected market narrative with lifecycle metadata."""

    name: str
    stage: str  # early | mainstream | exhausted
    confidence: float
    evidence: str
    related_tokens: list[str] = field(default_factory=list)
    sentiment: str = "neutral"
    momentum: str = "steady"


@dataclass
class NarrativeSnapshot:
    """Complete narrative landscape at a point in time."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    narratives: list[NarrativeSignal] = field(default_factory=list)
    dominant_narrative: str = ""
    contrarian_signal: str = ""

    @property
    def early_narratives(self) -> list[NarrativeSignal]:
        """Narratives in the early (most profitable) stage."""
        return [n for n in self.narratives if n.stage == "early"]

    @property
    def exhausted_narratives(self) -> list[NarrativeSignal]:
        """Narratives that may be topping out."""
        return [n for n in self.narratives if n.stage == "exhausted"]

    def format_for_prompt(self) -> str:
        """Format for inclusion in trading prompts."""
        if not self.narratives:
            return "No active narratives detected."

        lines = []
        for n in self.narratives:
            tokens = ", ".join(n.related_tokens[:3]) if n.related_tokens else "general"
            lines.append(
                f"- {n.name} [{n.stage}] ({n.sentiment}, {n.momentum}) "
                f"→ {tokens} (confidence: {n.confidence:.0%})"
            )

        if self.dominant_narrative:
            lines.append(f"\nDominant: {self.dominant_narrative}")
        if self.contrarian_signal:
            lines.append(f"⚠ Contrarian: {self.contrarian_signal}")

        return "\n".join(lines)


class NarrativePipeline:
    """Detects and classifies crypto market narratives using LLM analysis."""

    def __init__(self, llm_client: LLMClient, model: str = "gemini/gemini-2.0-flash") -> None:
        self._llm = llm_client
        self._model = model  # Use cheap model — narrative detection runs frequently
        self._last_snapshot: NarrativeSnapshot | None = None

    async def detect(self, headlines: list[str]) -> NarrativeSnapshot:
        """Analyze headlines to detect active narratives.

        Args:
            headlines: Recent news headlines and social media posts.
        """
        if not headlines:
            logger.info("narrative_skip", reason="no headlines")
            return NarrativeSnapshot()

        # Limit to recent 30 headlines to manage token usage
        recent = headlines[:30]
        headlines_text = "\n".join(f"- {h}" for h in recent)
        categories_text = "\n".join(f"- {c}" for c in NARRATIVE_CATEGORIES)

        prompt = NARRATIVE_DETECTION_PROMPT.format(
            headlines=headlines_text,
            categories=categories_text,
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a crypto narrative analyst. "
                        "Detect market narratives from news and social data. "
                        "Output ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            result = await self._llm.complete_json(messages, model=self._model)

            narratives = []
            for n in result.get("narratives", []):
                narratives.append(NarrativeSignal(
                    name=n.get("name", "unknown"),
                    stage=n.get("stage", "mainstream"),
                    confidence=float(n.get("confidence", 0.5)),
                    evidence=n.get("evidence", ""),
                    related_tokens=n.get("related_tokens", []),
                    sentiment=n.get("sentiment", "neutral"),
                    momentum=n.get("momentum", "steady"),
                ))

            snapshot = NarrativeSnapshot(
                narratives=narratives,
                dominant_narrative=result.get("dominant_narrative", ""),
                contrarian_signal=result.get("contrarian_signal", ""),
            )

            self._last_snapshot = snapshot

            logger.info(
                "narratives_detected",
                count=len(narratives),
                dominant=snapshot.dominant_narrative,
                early=[n.name for n in snapshot.early_narratives],
            )

            return snapshot

        except Exception as e:
            logger.error("narrative_detection_failed", error=str(e))
            return NarrativeSnapshot()

    @property
    def last_snapshot(self) -> NarrativeSnapshot | None:
        return self._last_snapshot
