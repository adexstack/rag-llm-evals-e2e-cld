"""
Tests for generation quality: Standard default metrics
    "answer_relevancy"
    "context_precision"
    "faithfulness"
    "context_recall"
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
    "get_standard_metrics_sample",
    load_test_data("testdata/standard_metrics.json"),
    indirect=True,
)
async def test_all_standard_metrics(
    llm_wrapper,
    embeddings_wrapper,
    get_standard_metrics_sample: SingleTurnSample,
    settings,
    results_path,
) -> None:
    """Run all four core metrics concurrently and persist results to CSV."""
    s = get_standard_metrics_sample

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