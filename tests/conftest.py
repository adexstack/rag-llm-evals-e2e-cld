"""Shared pytest fixtures for the RAG evaluation test suite.

Fixture scoping strategy
------------------------
session  - expensive objects that are safe to reuse (LLM / embeddings clients)
function - sample objects that are built per-test from parametrized test data
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from openai import AsyncOpenAI
from ragas import MultiTurnSample, SingleTurnSample
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory

from rag_evals.client import RagClient
from rag_evals.config import get_settings
from rag_evals.samples import (
    build_multi_turn_sample_live,
    build_multi_turn_sample_static,
    build_single_turn_from_data,
    build_single_turn_sample,
    load_sample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session-scoped infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def settings():
    # To test ConfigurationError or env-var overrides in a future test_config.py,
    # call get_settings.cache_clear() before and after via monkeypatch.setenv /
    # monkeypatch.delenv. The lru_cache persists for the process lifetime otherwise.
    return get_settings()


@pytest.fixture(scope="session")
def async_openai_client(settings):
    """Single AsyncOpenAI client reused for the whole test session."""
    return AsyncOpenAI(api_key=settings.openai_api_key)


@pytest.fixture(scope="session")
def llm_wrapper(async_openai_client, settings):
    """Ragas-compatible LLM wrapper (session-scoped — expensive to create)."""
    return llm_factory(model=settings.llm_model, client=async_openai_client)


@pytest.fixture(scope="session")
def embeddings_wrapper(async_openai_client, settings):
    """Ragas-compatible embeddings wrapper (session-scoped)."""
    return OpenAIEmbeddings(client=async_openai_client, model=settings.embedding_model)


@pytest.fixture(scope="session")
def rag_client(settings) -> RagClient:
    """Single RagClient reused for the whole test session."""
    return RagClient(base_url=settings.rag_api_base_url, timeout=settings.api_timeout_seconds)


@pytest.fixture(scope="session")
def results_path() -> Path:
    """Filesystem path for persisted metric results. Override per-test for isolation."""
    return Path("reports/ragas_results.csv")


# ---------------------------------------------------------------------------
# Sample fixtures — built per test from parametrized data
# ---------------------------------------------------------------------------


@pytest.fixture
def get_precision_sample(request, rag_client) -> SingleTurnSample:
    return build_single_turn_sample(request.param, rag_client, include_reference=False)


@pytest.fixture
def get_recall_sample(request, rag_client) -> SingleTurnSample:
    return build_single_turn_sample(request.param, rag_client, include_reference=True)


@pytest.fixture
def get_faithfulness_sample(request, rag_client) -> SingleTurnSample:
    return build_single_turn_sample(request.param, rag_client, include_reference=False)


@pytest.fixture
def get_relevance_fact_sample(request, rag_client) -> SingleTurnSample:
    return build_single_turn_sample(request.param, rag_client, include_reference=True)

@pytest.fixture
def get_standard_metrics_sample(request, rag_client) -> SingleTurnSample:
    return build_single_turn_sample(request.param, rag_client, include_reference=True)

@pytest.fixture
def get_single_turn_sample() -> SingleTurnSample:
    return build_single_turn_from_data(load_sample("testdata/single_turn_sample.json"))


@pytest.fixture
def get_topic_data_static() -> MultiTurnSample:
    return build_multi_turn_sample_static(load_sample("testdata/multi_turn_static.json"))


@pytest.fixture
def get_topic_data_live(request, rag_client) -> MultiTurnSample:
    if not request.config.getoption("--live"):
        pytest.skip("pass --live to run tests that make live calls via fixtures")
    return build_multi_turn_sample_live(load_sample("testdata/multi_turn_live.json"), rag_client)
