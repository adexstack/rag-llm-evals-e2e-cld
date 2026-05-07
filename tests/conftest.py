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
from ragas.messages import AIMessage, HumanMessage

from rag_evals.client import RagClient
from rag_evals.config import get_settings
from rag_evals.samples import build_single_turn_sample

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
def get_relevance_factual_sample(request, rag_client) -> SingleTurnSample:
    return build_single_turn_sample(request.param, rag_client, include_reference=True)


@pytest.fixture
def get_single_turn_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Europe and it is part of France.",
        reference="The Eiffel Tower is located in Paris.",
    )


@pytest.fixture
def get_topic_data_static() -> MultiTurnSample:
    """Hardcoded multi-turn sample for deterministic topic-adherence tests."""
    conversation = [
        HumanMessage(content="How many articles are there in the selenium webdriver python course"),
        AIMessage(content="There are 23 articles in the Selenium Webdriver Python course"),
        HumanMessage(content="How many downloadable resources are there in this course?"),
        AIMessage(content="There are 9 downloadable resources in the course."),
    ]
    reference = [
        "The AI should:\n"
        "1. Give results related to the selenium webdriver python course\n"
        "2. There are 23 articles and 9 downloadable resources in the course"
    ]
    return MultiTurnSample(user_input=conversation, reference_topics=reference)


@pytest.fixture
def get_topic_data_live(request, rag_client) -> MultiTurnSample:
    """Multi-turn sample built from live RAG API responses."""
    if not request.config.getoption("--live"):
        pytest.skip("pass --live to run tests that make live calls via fixtures")
    questions = [
        "How many articles are there in the selenium webdriver python course",
        "How many downloadable resources are there in this course?",
    ]
    conversation: list[HumanMessage | AIMessage] = []
    for question in questions:
        response_dict = rag_client.ask(question)
        conversation.append(HumanMessage(content=question))
        conversation.append(AIMessage(content=response_dict["answer"]))
        logger.info("Live API answer for %r: %s", question, response_dict["answer"][:80])

    reference = [
        "The AI should:\n"
        "1. Give results related to the selenium webdriver python course\n"
        "2. There are 23 articles and 9 downloadable resources in the course"
    ]
    return MultiTurnSample(user_input=conversation, reference_topics=reference)
