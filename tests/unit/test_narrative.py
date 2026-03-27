"""Tests for narrative detection pipeline and evolution node."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.narrative import NarrativePipeline, NarrativeSignal, NarrativeSnapshot


class TestNarrativeSnapshot:
    """Test NarrativeSnapshot data model."""

    def test_early_narratives(self):
        snap = NarrativeSnapshot(narratives=[
            NarrativeSignal(name="AI tokens", stage="early", confidence=0.8, evidence="growing mentions"),
            NarrativeSignal(name="Memecoins", stage="exhausted", confidence=0.9, evidence="peak hype"),
            NarrativeSignal(name="DePIN", stage="early", confidence=0.6, evidence="new projects"),
        ])
        early = snap.early_narratives
        assert len(early) == 2
        assert early[0].name == "AI tokens"
        assert early[1].name == "DePIN"

    def test_exhausted_narratives(self):
        snap = NarrativeSnapshot(narratives=[
            NarrativeSignal(name="Memecoins", stage="exhausted", confidence=0.9, evidence="peak"),
        ])
        assert len(snap.exhausted_narratives) == 1
        assert snap.exhausted_narratives[0].name == "Memecoins"

    def test_empty_snapshot(self):
        snap = NarrativeSnapshot()
        assert len(snap.narratives) == 0
        assert snap.early_narratives == []
        assert snap.exhausted_narratives == []

    def test_format_for_prompt_empty(self):
        snap = NarrativeSnapshot()
        assert snap.format_for_prompt() == "No active narratives detected."

    def test_format_for_prompt_with_data(self):
        snap = NarrativeSnapshot(
            narratives=[
                NarrativeSignal(
                    name="AI tokens",
                    stage="early",
                    confidence=0.8,
                    evidence="test",
                    related_tokens=["FET", "RNDR"],
                    sentiment="bullish",
                    momentum="accelerating",
                ),
            ],
            dominant_narrative="AI tokens",
            contrarian_signal="Memecoins overheated",
        )
        text = snap.format_for_prompt()
        assert "AI tokens" in text
        assert "early" in text
        assert "FET" in text
        assert "Dominant" in text
        assert "Contrarian" in text


class TestNarrativePipeline:
    """Test the LLM-powered narrative detection."""

    @pytest.mark.asyncio
    async def test_detect_with_no_headlines(self):
        mock_llm = MagicMock()
        pipeline = NarrativePipeline(llm_client=mock_llm)
        result = await pipeline.detect([])
        assert len(result.narratives) == 0

    @pytest.mark.asyncio
    async def test_detect_parses_llm_response(self):
        mock_llm = MagicMock()
        mock_llm.complete_json = AsyncMock(return_value={
            "narratives": [
                {
                    "name": "AI tokens",
                    "stage": "early",
                    "confidence": 0.85,
                    "evidence": "FET and RNDR surging",
                    "related_tokens": ["FET", "RNDR", "OCEAN"],
                    "sentiment": "bullish",
                    "momentum": "accelerating",
                },
                {
                    "name": "Memecoins",
                    "stage": "exhausted",
                    "confidence": 0.7,
                    "evidence": "DOGE volume declining",
                    "related_tokens": ["DOGE", "SHIB"],
                    "sentiment": "bearish",
                    "momentum": "decelerating",
                },
            ],
            "dominant_narrative": "AI tokens",
            "contrarian_signal": "Memecoins may reverse",
        })

        pipeline = NarrativePipeline(llm_client=mock_llm)
        result = await pipeline.detect(["FET hits new high", "AI narrative growing"])

        assert len(result.narratives) == 2
        assert result.narratives[0].name == "AI tokens"
        assert result.narratives[0].stage == "early"
        assert result.narratives[0].confidence == 0.85
        assert result.dominant_narrative == "AI tokens"
        assert pipeline.last_snapshot is result

    @pytest.mark.asyncio
    async def test_detect_handles_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.complete_json = AsyncMock(side_effect=Exception("API down"))

        pipeline = NarrativePipeline(llm_client=mock_llm)
        result = await pipeline.detect(["some headline"])
        assert len(result.narratives) == 0

    @pytest.mark.asyncio
    async def test_detect_limits_headlines(self):
        mock_llm = MagicMock()
        mock_llm.complete_json = AsyncMock(return_value={
            "narratives": [],
            "dominant_narrative": "",
            "contrarian_signal": "",
        })

        pipeline = NarrativePipeline(llm_client=mock_llm)
        headlines = [f"Headline {i}" for i in range(50)]
        await pipeline.detect(headlines)

        # Check prompt only includes first 30
        call_args = mock_llm.complete_json.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "Headline 29" in user_msg
        assert "Headline 30" not in user_msg
