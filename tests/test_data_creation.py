"""Tests for synthetic test-data generation via Ragas TestsetGenerator."""
from __future__ import annotations

import logging

import pytest

from rag_evals.samples import load_documents

logger = logging.getLogger(__name__)

_MOCK_DOCS, _EXPECTED_SAMPLE_COUNT = load_documents("testdata/mock_docs.json")


@pytest.mark.data_creation
@pytest.mark.slow
@pytest.mark.asyncio
async def test_testset_generation(llm_wrapper, embeddings_wrapper) -> None:
    """
    Verify that TestsetGenerator produces the expected number of samples.
    In Ragas, this test ensures that the TestsetGenerator creates exactly the intended number of 
    evaluation samples based on the configured parameters.
    It validates that generation logic (e.g., chunking, sampling strategy) is functioning correctly.
    This helps prevent under- or over-generation, which can skew evaluation results.
    Passing the test confirms dataset consistency and reliability for downstream LLM metric evaluation.
    """
    from ragas.testset import TestsetGenerator  # local import — keeps 'Test*' name off module scope

    generator = TestsetGenerator(llm=llm_wrapper, embedding_model=embeddings_wrapper)
    # generate_with_langchain_docs returns EvaluationDataset at runtime; ragas stubs mis-annotate it as Executor
    dataset = generator.generate_with_langchain_docs(_MOCK_DOCS, testset_size=_EXPECTED_SAMPLE_COUNT)  # type: ignore[assignment]
    samples = dataset.to_list()  # type: ignore[attr-defined]

    logger.info("Generated %d test samples", len(samples))

    assert len(samples) == _EXPECTED_SAMPLE_COUNT, (
        f"Expected {_EXPECTED_SAMPLE_COUNT} samples, got {len(samples)}"
    )
    # Ragas >=0.4 uses 'user_input' / 'reference' in to_list() output, not 'question' / 'answer'
    for i, sample in enumerate(samples):
        assert "user_input" in sample, f"Sample {i} is missing 'user_input'"
        assert "reference" in sample, f"Sample {i} is missing 'reference'"
        assert sample["user_input"], f"Sample {i} has an empty user_input"
