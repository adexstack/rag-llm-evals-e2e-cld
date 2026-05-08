"""Centralised configuration with a three-layer priority stack:

    env vars  >  config.json  >  built-in fallbacks

Users can change any non-secret setting by editing config.json at the
project root.  CI/CD can override individual values via environment
variables without touching any file.  Secrets (API keys) are env-var only
and are never read from config.json.

Threshold behaviour
-------------------
Set a threshold to "" (empty string) in config.json or as an env var to
disable the assertion for that specific metric.  Set assert_test_threshold
to false to skip ALL threshold assertions at once.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .exceptions import ConfigurationError

load_dotenv()

_CONFIG_FILE = Path(__file__).parent.parent.parent / "config.json"


def _load_config_file() -> dict[str, Any]:
    """Return the contents of config.json, or {} if the file is absent."""
    if not _CONFIG_FILE.exists():
        return {}
    with _CONFIG_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


def _env(key: str, file_value: Any, fallback: Any) -> str:
    """Resolve a single string setting: env var > config.json value > fallback."""
    return os.getenv(key, str(file_value if file_value is not None else fallback))


def _threshold_value(env_key: str, file_value: Any, fallback: float) -> float | None:
    """Resolve a threshold value.

    Returns None (skip assertion) when:
    - the env var is set to an empty string, or
    - the config.json value is an empty string "".
    Otherwise returns the resolved float.
    """
    env_raw = os.getenv(env_key)
    if env_raw is not None:
        return None if env_raw == "" else float(env_raw)
    if file_value == "":
        return None
    return float(file_value) if file_value is not None else float(fallback)


def _parse_bool(value: str) -> bool:
    return value.lower() not in ("false", "0", "no", "off")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    ragas_app_token: str
    rag_api_base_url: str
    llm_model: str
    embedding_model: str
    api_timeout_seconds: float
    assert_test_threshold: bool
    score_thresholds: dict[str, float | None]

    def threshold_for(self, metric: str) -> float | None:
        """Return the threshold for *metric*, or None if assertions are disabled.

        Returns None when assert_test_threshold is false OR the metric's
        value is empty in config.json / its env var.
        """
        if not self.assert_test_threshold:
            return None
        return self.score_thresholds.get(metric)

    @classmethod
    def from_env(cls) -> "Settings":
        cfg = _load_config_file()
        thresholds_cfg = cfg.get("score_thresholds", {})

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or export it in your shell."
            )

        assert_flag = _parse_bool(
            os.getenv("ASSERT_TEST_THRESHOLD", str(cfg.get("assert_test_threshold", True)))
        )

        return cls(
            openai_api_key=api_key,
            ragas_app_token=os.getenv("RAGAS_APP_TOKEN", ""),
            rag_api_base_url=_env("RAG_API_BASE_URL", cfg.get("rag_api_base_url"), "https://rahulshettyacademy.com/rag-llm"),
            llm_model=_env("LLM_MODEL", cfg.get("llm_model"), "gpt-4o"),
            embedding_model=_env("EMBEDDING_MODEL", cfg.get("embedding_model"), "text-embedding-3-small"),
            api_timeout_seconds=float(_env("API_TIMEOUT_SECONDS", cfg.get("api_timeout_seconds"), 30)),
            assert_test_threshold=assert_flag,
            score_thresholds=_build_thresholds(thresholds_cfg),
        )


def _build_thresholds(cfg: dict[str, Any]) -> dict[str, float | None]:
    return {
        "context_precision": _threshold_value("THRESHOLD_CONTEXT_PRECISION", cfg.get("context_precision"), 0.8),
        "context_recall":    _threshold_value("THRESHOLD_CONTEXT_RECALL",    cfg.get("context_recall"),    0.7),
        "faithfulness":      _threshold_value("THRESHOLD_FAITHFULNESS",      cfg.get("faithfulness"),      0.8),
        "answer_relevancy":  _threshold_value("THRESHOLD_ANSWER_RELEVANCY",  cfg.get("answer_relevancy"),  0.8),
        "topic_adherence":   _threshold_value("THRESHOLD_TOPIC_ADHERENCE",   cfg.get("topic_adherence"),   0.8),
        "rubric_score_min":  _threshold_value("THRESHOLD_RUBRIC_SCORE_MIN",  cfg.get("rubric_score_min"),  2.0),
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings.from_env()
