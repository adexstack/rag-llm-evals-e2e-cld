"""Tests for generation quality: Faithfulness, Answer Relevancy, Answer Correctness.

Also includes the all-metrics aggregate test that writes results to CSV.
"""
from __future__ import annotations

import asyncio
import logging

import pytest
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from rag_evals.reporting import save_results
from rag_evals.samples import load_test_data

logger = logging.getLogger(__name__)


@pytest.mark.generation
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_faithfulness_sample",
    load_test_data("testdata/faithfulness_data.json"),
    indirect=True,
)
async def test_faithfulness(
    llm_wrapper,
    get_faithfulness_sample: SingleTurnSample,
    settings,
) -> None:
    """
    Faithfulness metric measures the factual consistency of the generated answer against 
    the given context.
    It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. 
    Higher the better. 
    """
    metric = Faithfulness(llm=llm_wrapper)
    score = await metric.ascore(
        user_input=get_faithfulness_sample.user_input,
        response=get_faithfulness_sample.response,
        retrieved_contexts=get_faithfulness_sample.retrieved_contexts,
    )
    threshold = settings.score_thresholds["faithfulness"]
    logger.info("faithfulness=%.4f  threshold=%.2f", score, threshold)
    assert score > threshold, f"Faithfulness {float(score):.4f} is below threshold {threshold}"


@pytest.mark.generation
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_relevance_factual_sample",
    load_test_data("testdata/relevance_fact.json"),
    indirect=True,
)
async def test_relevancy_metrics_initialise(
    llm_wrapper,
    embeddings_wrapper,
    get_relevance_factual_sample: SingleTurnSample,
    settings,
) -> None:
    """
    Answer Relevancy metric:
    This metric focuses on assessing how pertinent the generated answer is to the given 
    prompt. A lower score is assigned to answers that are incomplete or contain redundant information 
    and higher scores indicate better relevancy. 
    Assessment of answer relevance does not consider factuality but instead penalizes cases where the 
    answer lacks completeness or contains redundant details. To calculate this score, the LLM is prompted
    to generate an appropriate question for the generated answer multiple times, and the mean cosine 
    similarity between these generated questions and the original question is measured.

    Example:
    Question: Where is France and what is it's capital?
    Low relevance answer: France is in western Europe.
    High relevance answer: France is in western Europe and Paris is its capital.

    Factual Correctness:
    Factual Correctness is a metric that compares and evaluates the factual accuracy of the generated
    response with the reference (ground truth/expected value). This metric is used to determine the 
    extent to which the generated response aligns with the reference. The factual correctness score 
    ranges from 0 to 1, with higher values indicating better performance. 
    """
    answer_relevancy = AnswerRelevancy(llm=llm_wrapper, embeddings=embeddings_wrapper)
    answer_correctness = AnswerCorrectness(llm=llm_wrapper, embeddings=embeddings_wrapper)

    assert answer_relevancy is not None
    assert answer_correctness is not None

    s = get_relevance_factual_sample
    score = await answer_relevancy.ascore(
        user_input=s.user_input,
        response=s.response,
    )
    threshold = settings.score_thresholds["answer_relevancy"]
    logger.info("answer_relevancy=%.4f  threshold=%.2f", score, threshold)
    assert score > threshold, f"answer_relevancy={score:.4f} is below threshold {threshold}"


@pytest.mark.generation
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_relevance_factual_sample",
    load_test_data("testdata/relevance_fact.json"),
    indirect=True,
)
async def test_all_standard_metrics(
    llm_wrapper,
    embeddings_wrapper,
    get_relevance_factual_sample: SingleTurnSample,
    settings,
    results_path,
) -> None:
    """Run all four core metrics concurrently and persist results to CSV."""
    s = get_relevance_factual_sample

    answer_relevancy = AnswerRelevancy(llm=llm_wrapper, embeddings=embeddings_wrapper)
    context_precision = ContextPrecisionWithoutReference(llm=llm_wrapper)
    faithfulness = Faithfulness(llm=llm_wrapper)
    context_recall = ContextRecall(llm=llm_wrapper)

    ar, cp, f, cr = await asyncio.gather(
        answer_relevancy.ascore(
            user_input=s.user_input, 
            response=s.response),
        context_precision.ascore(
            user_input=s.user_input,
            response=s.response,
            retrieved_contexts=s.retrieved_contexts,
        ),
        faithfulness.ascore(
            user_input=s.user_input,
            response=s.response,
            retrieved_contexts=s.retrieved_contexts,
        ),
        context_recall.ascore(
            user_input=s.user_input,
            retrieved_contexts=s.retrieved_contexts,
            reference=s.reference,
        ),
    )

    results = {
        "answer_relevancy": ar.value,
        "context_precision": cp.value,
        "faithfulness": f.value,
        "context_recall": cr.value,
    }
    for metric, value in results.items():
        logger.info("%s=%.4f", metric, value)

    save_results(results, results_path)

    thresholds = settings.score_thresholds
    assert ar.value > thresholds["answer_relevancy"], f"answer_relevancy={ar.value:.4f}"
    assert cp.value > thresholds["context_precision"], f"context_precision={cp.value:.4f}"
    assert f.value > thresholds["faithfulness"], f"faithfulness={f.value:.4f}"
    assert cr.value > thresholds["context_recall"], f"context_recall={cr.value:.4f}"