"""Helpers that turn raw API responses into Ragas sample objects."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ragas import SingleTurnSample

from . import client

logger = logging.getLogger(__name__)


def load_test_data(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON test-data from *path* and return it as a list of dicts."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    with file_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {file_path}, got {type(data).__name__}")
    logger.debug("Loaded %d records from %s", len(data), file_path)
    return data


def build_single_turn_sample(
    test_data: dict[str, Any],
    *,
    include_reference: bool = False,
) -> SingleTurnSample:
    """Fetch the RAG response for *test_data* and wrap it in a SingleTurnSample.

    Args:
        test_data: dict with at minimum a ``"question"`` key; optionally
                   ``"reference"`` when ``include_reference=True``.
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
