"""Tests for rubric-based scoring."""
from __future__ import annotations

import logging

import pytest
from ragas import SingleTurnSample
from ragas.metrics.collections import RubricsScoreWithReference

from rag_evals.config import get_settings

logger = logging.getLogger(__name__)

_RUBRICS = {
    "score1_description": (
        "The response is incorrect, irrelevant, or does not align with the ground truth."
    ),
    "score2_description": (
        "The response partially matches the ground truth but includes significant "
        "errors, omissions, or irrelevant information."
    ),
    "score3_description": (
        "The response generally aligns with the ground truth but may lack detail, "
        "clarity, or have minor inaccuracies."
    ),
    "score4_description": (
        "The response is mostly accurate and aligns well with the ground truth, "
        "with only minor issues or missing details."
    ),
    "score5_description": (
        "The response is fully accurate, aligns completely with the ground truth, "
        "and is clear and detailed."
    ),
}


@pytest.mark.asyncio
async def test_rubric_score(
    llm_wrapper,
    get_single_turn_sample: SingleTurnSample,
    settings,
) -> None:
    """
    This metric that is used to evaluate response. The rubric consists of descriptions
    for each score, typically ranging from 1 to 10. The response here is evaluation based 
    on score_descriptions and ground truth.
    """
    metric = RubricsScoreWithReference(rubrics=_RUBRICS, llm=llm_wrapper)
    score = await metric.ascore(
        user_input=get_single_turn_sample.user_input,
        response=get_single_turn_sample.response,
        reference=get_single_turn_sample.reference,
    )
    min_score = settings.score_thresholds["rubric_score_min"]
    logger.info("rubric_score=%.1f  min_expected=%.1f", score, min_score)
    assert 1 <= score <= 5, f"Rubric score {score} is outside valid range [1, 5]"
    assert score >= min_score, f"Rubric score {score} is below minimum {min_score}"
