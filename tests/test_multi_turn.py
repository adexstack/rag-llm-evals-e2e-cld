"""
Tests for multi-turn conversation quality: Topic Adherence.
Topic Adherence in Ragas measures whether an LLM stays focused on the intended subject across multiple turns in a conversation.
It evaluates if each response remains relevant to the original query or evolving context without drifting off-topic.
The metric typically compares responses against the conversation context using semantic similarity or reference grounding.
High scores indicate coherent, context-aware dialogue, while low scores reveal topic drift or irrelevant responses.
"""
from __future__ import annotations

import logging

import pytest
from ragas import MultiTurnSample
from ragas.metrics._topic_adherence import TopicAdherenceScore

from rag_evals.config import get_settings

logger = logging.getLogger(__name__)


@pytest.mark.multi_turn
@pytest.mark.slow
@pytest.mark.asyncio
async def test_topic_adherence_static(
    llm_wrapper,
    get_topic_data_static: MultiTurnSample,
    settings,
) -> None:
    """Topic adherence test using hardcoded, deterministic conversation."""
    metric = TopicAdherenceScore(llm=llm_wrapper)
    score = await metric.multi_turn_ascore(get_topic_data_static)
    threshold = settings.threshold_for("topic_adherence")
    logger.info("topic_adherence (static)=%.4f  threshold=%s", score, threshold)
    if threshold is not None:
        assert float(score) > threshold, f"Topic adherence {float(score):.4f} is below threshold {threshold}"


@pytest.mark.multi_turn
@pytest.mark.slow
@pytest.mark.asyncio
async def test_topic_adherence_live(
    llm_wrapper,
    get_topic_data_live: MultiTurnSample,
    settings,
) -> None:
    """Topic adherence test using live RAG API responses."""
    metric = TopicAdherenceScore(llm=llm_wrapper)
    score = await metric.multi_turn_ascore(get_topic_data_live)
    threshold = settings.threshold_for("topic_adherence")
    logger.info("topic_adherence (live)=%.4f  threshold=%s", score, threshold)
    if threshold is not None:
        assert float(score) > threshold, f"Topic adherence {float(score):.4f} is below threshold {threshold}"
