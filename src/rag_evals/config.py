"""Centralised configuration loaded from environment variables.

All settings are read once at import time.  Tests and application code
import from here instead of scattering os.getenv() calls everywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

from .exceptions import ConfigurationError

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    ragas_app_token: str
    rag_api_base_url: str
    llm_model: str
    embedding_model: str
    api_timeout_seconds: float
    score_thresholds: dict[str, float]

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or export it in your shell."
            )
        return cls(
            openai_api_key=api_key,
            ragas_app_token=os.getenv("RAGAS_APP_TOKEN", ""),
            rag_api_base_url=os.getenv(
                "RAG_API_BASE_URL",
                "https://rahulshettyacademy.com/rag-llm",
            ),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", "30")),
            score_thresholds=_default_thresholds(),
        )


def _default_thresholds() -> dict[str, float]:
    return {
        "context_precision": float(os.getenv("THRESHOLD_CONTEXT_PRECISION", "0.8")),
        "context_recall": float(os.getenv("THRESHOLD_CONTEXT_RECALL", "0.7")),
        "faithfulness": float(os.getenv("THRESHOLD_FAITHFULNESS", "0.8")),
        "answer_relevancy": float(os.getenv("THRESHOLD_ANSWER_RELEVANCY", "0.8")),
        "topic_adherence": float(os.getenv("THRESHOLD_TOPIC_ADHERENCE", "0.8")),
        "rubric_score_min": float(os.getenv("THRESHOLD_RUBRIC_SCORE_MIN", "2.0")),
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings.from_env()
