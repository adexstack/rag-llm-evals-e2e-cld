"""Helpers that turn raw API responses into Ragas sample objects."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from ragas import MultiTurnSample, SingleTurnSample
from ragas.messages import AIMessage, HumanMessage

from .client import RagClient

logger = logging.getLogger(__name__)

# Resolves testdata/ paths relative to the project root regardless of CWD.
# src/rag_evals/samples.py → src/rag_evals/ → src/ → project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def load_test_data(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON test-data from *path* and return it as a list of dicts."""
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Test data file not found: {resolved}")
    with resolved.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {resolved}, got {type(data).__name__}")
    logger.debug("Loaded %d records from %s", len(data), resolved)
    return data


def load_sample(path: str | Path) -> dict[str, Any]:
    """Load a single JSON object from *path*."""
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Sample data file not found: {resolved}")
    with resolved.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {resolved}, got {type(data).__name__}")
    return data


def load_documents(path: str | Path) -> tuple[list[Document], int]:
    """Load LangChain Documents and expected sample count from *path*."""
    data = load_sample(path)
    docs = [
        Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
        for d in data["documents"]
    ]
    return docs, int(data["expected_sample_count"])


def build_single_turn_from_data(data: dict[str, Any]) -> SingleTurnSample:
    """Build a SingleTurnSample from a pre-fetched data dict (no API call)."""
    return SingleTurnSample(
        user_input=data["user_input"],
        response=data["response"],
        reference=data.get("reference"),
    )


def _parse_message(msg: dict[str, str]) -> HumanMessage | AIMessage:
    role = msg["role"].lower()
    if role == "human":
        return HumanMessage(content=msg["content"])
    if role == "ai":
        return AIMessage(content=msg["content"])
    raise ValueError(f"Unknown message role: {role!r}")


def build_multi_turn_sample_static(data: dict[str, Any]) -> MultiTurnSample:
    """Build a MultiTurnSample from static conversation data (no API call)."""
    conversation = [_parse_message(m) for m in data["conversation"]]
    return MultiTurnSample(user_input=conversation, reference_topics=data["reference_topics"])


def build_multi_turn_sample_live(data: dict[str, Any], client: RagClient) -> MultiTurnSample:
    """Build a MultiTurnSample by calling the RAG API for each question in *data*."""
    conversation: list[HumanMessage | AIMessage] = []
    for question in data["questions"]:
        response_dict = client.ask(question)
        conversation.append(HumanMessage(content=question))
        conversation.append(AIMessage(content=response_dict["answer"]))
        logger.info("Live API answer for %r: %s", question, response_dict["answer"][:80])
    return MultiTurnSample(user_input=conversation, reference_topics=data["reference_topics"])


def build_single_turn_sample(
    test_data: dict[str, Any],
    client: RagClient,
    *,
    include_reference: bool = False,
) -> SingleTurnSample:
    """Fetch the RAG response for *test_data* and wrap it in a SingleTurnSample.

    Args:
        test_data: dict with at minimum a ``"question"`` key; optionally
                   ``"reference"`` when ``include_reference=True``.
        client: the RagClient instance to use for the API call.
        include_reference: when True the sample's ``reference`` field is
                           populated from ``test_data["reference"]``.

    Raises:
        KeyError: if *test_data* is missing required keys.
        RagAPIError / RagAPITimeoutError: propagated from the client layer.
    """
    question: str = test_data["question"]
    response_dict = client.ask(question)

    logger.info("Fetched RAG response for question=%r", question[:80])

    kwargs: dict[str, Any] = {
        "user_input": question,
        "response": response_dict.get("answer", ""),
        "retrieved_contexts": [
            doc["page_content"] for doc in response_dict.get("retrieved_docs", [])
        ],
    }
    if include_reference:
        kwargs["reference"] = test_data["reference"]

    return SingleTurnSample(**kwargs)
