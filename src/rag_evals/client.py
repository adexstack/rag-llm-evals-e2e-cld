"""HTTP client for the RAG backend API."""
from __future__ import annotations

import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import RagAPIError, RagAPITimeoutError

logger = logging.getLogger(__name__)

_RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist={500, 502, 503, 504},
    allowed_methods={"POST"},
)


def _build_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY_STRATEGY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class RagClient:
    """HTTP client for the RAG /ask endpoint."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._session = session or _build_session()

    def ask(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Call the RAG /ask endpoint and return the parsed JSON response.

        Raises:
            RagAPITimeoutError: if the request exceeds the configured timeout.
            RagAPIError: if the server returns a non-2xx status code.
        """
        url = f"{self._base_url}/ask"
        payload = {"question": question, "chat_history": chat_history or []}

        logger.debug("POST %s  question=%r", url, question)
        try:
            response = self._session.post(url, json=payload, timeout=self._timeout)
        except requests.Timeout as exc:
            raise RagAPITimeoutError(
                f"Request to {url} timed out after {self._timeout}s"
            ) from exc
        except requests.ConnectionError as exc:
            raise RagAPIError(f"Connection error reaching {url}: {exc}") from exc

        if not response.ok:
            raise RagAPIError(
                f"RAG API returned {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        data: dict[str, Any] = response.json()
        logger.debug("Response keys: %s", list(data.keys()))
        return data
