"""Tests for retrieval quality: Context Precision and Context Recall.
How They Complement Each Other
High Recall + Low Precision: Many relevant documents are retrieved, but there’s also a lot of noise (irrelevant documents).
High Precision + Low Recall: Most retrieved documents are relevant, but some important ones are missing.
High Recall + High Precision: Ideal case—comprehensive and accurate retrieval.
In practice:
Recall is prioritized when missing relevant documents is costly (e.g., medical research, legal discovery).
Precision is prioritized when irrelevant information can overwhelm the user (e.g., search engine queries).
"""

from __future__ import annotations

import logging

import pytest
from ragas import SingleTurnSample
from ragas.metrics.collections import ContextPrecisionWithoutReference, ContextRecall

from rag_evals.config import get_settings
from rag_evals.samples import load_test_data

logger = logging.getLogger(__name__)


@pytest.mark.retrieval
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_precision_sample",
    load_test_data("testdata/precision_data.json"),
    indirect=True,
)
async def test_context_precision(
    llm_wrapper,
    get_precision_sample: SingleTurnSample,
    settings,
) -> None:
    """
    Context Precision:
    Measures the proportion of retrieved documents that are relevant to the query out of all the documents retrieved.
    Context Precision = Number of relevant documents retrieved / Total number of documents retrieved
    Higher the better.
    """
    metric = ContextPrecisionWithoutReference(llm=llm_wrapper)
    score = await metric.ascore(
        user_input=get_precision_sample.user_input,
        response=get_precision_sample.response,
        retrieved_contexts=get_precision_sample.retrieved_contexts,
    )
    threshold = settings.threshold_for("context_precision")
    logger.info("context_precision=%.4f  threshold=%s", score, threshold)
    if threshold is not None:
        assert float(score) > threshold, f"Context precision {float(score):.4f} is below threshold {threshold}"


@pytest.mark.retrieval
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "get_recall_sample",
    load_test_data("testdata/recall_data.json"),
    indirect=True,
)
async def test_context_recall(
    llm_wrapper,
    get_recall_sample: SingleTurnSample,
    settings,
) -> None:
    """
    Context Recall:
    Measures the proportion of relevant documents that are retrieved out of all the relevant documents.
    Context Recall = Number of relevant documents retrieved / Total number of relevant documents
    Higher the better.
    """
    metric = ContextRecall(llm=llm_wrapper)
    score = await metric.ascore(
        user_input=get_recall_sample.user_input,
        retrieved_contexts=get_recall_sample.retrieved_contexts,
        reference=get_recall_sample.reference,
    )
    threshold = settings.threshold_for("context_recall")
    logger.info("context_recall=%.4f  threshold=%s", score, threshold)
    if threshold is not None:
        assert float(score) > threshold, f"Context recall {float(score):.4f} is below threshold {threshold}"
