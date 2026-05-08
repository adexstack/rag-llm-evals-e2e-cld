# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/test_generation.py -v

# Run a single test by name
python -m pytest tests/test_generation.py::test_faithfulness -v

# Run tests by marker (skip slow API-heavy tests)
python -m pytest -m "not slow"
python -m pytest -m retrieval
python -m pytest -m generation

# Run with HTML report
python -m pytest --html=reports/test_report.html

# Generate results dashboard (reads reports/ragas_results.csv)
uv run python scripts/create_dashboard.py

# Launch Streamlit dashboard
uv run streamlit run scripts/view_results.py
```

**CRITICAL:** Always use `python -m pytest`, never bare `pytest`. The bare binary skips `.pth` editable-install resolution; `pythonpath = src` in `pytest.ini` only activates under `python -m`.

## Architecture

The package lives in `src/rag_evals/` (editable install via `pythonpath = src` in `pytest.ini`). Tests live in `tests/`. Test data (static JSON fixtures) lives in `testdata/`.

**Data flow per test:**
1. `testdata/*.json` → `load_test_data()` → parametrize decorator
2. Indirect fixture (`get_precision_sample`, etc.) → `build_single_turn_sample()` → calls live RAG API via `RagClient`
3. `SingleTurnSample` → Ragas metric `.ascore()` → score compared against threshold from `Settings`

**`src/rag_evals/` modules:**
- `config.py` — Frozen `Settings` dataclass loaded from env vars via `python-dotenv`. Singleton via `lru_cache`. `OPENAI_API_KEY` is required; all thresholds are overridable via `THRESHOLD_*` env vars.
- `client.py` — `RagClient` wraps the `/ask` endpoint with retry (3×, backoff 0.5, on 5xx), 30s timeout. Returns `{"answer": str, "retrieved_docs": [{"page_content": str}]}`.
- `samples.py` — `build_single_turn_sample()` calls `RagClient.ask()` and builds a `SingleTurnSample`. `load_test_data()` resolves paths relative to project root regardless of CWD.
- `reporting.py` — `save_results()` writes a single-row CSV to `reports/ragas_results.csv`.
- `exceptions.py` — `RagAPIError`, `RagAPITimeoutError`, `ConfigurationError`.

**Ragas metric return types:** `ascore()` returns a `MetricResult` object, not a plain `float`. It supports `%`-style formatting and `float()` conversion, but **not** f-string `:.4f` format specs. Always wrap in `float()` when formatting in assert messages: `f"score={float(score):.4f}"`.

**`TestsetGenerator` import:** Must be imported inside the test function body in `test_data_creation.py`. Importing at module scope causes a `PytestCollectionWarning` because the class name starts with `Test`.

## Test Markers

| Marker | What it covers |
|---|---|
| `retrieval` | `ContextPrecisionWithoutReference`, `ContextRecall` |
| `generation` | `Faithfulness`, `AnswerRelevancy`, `AnswerCorrectness` |
| `multi_turn` | `TopicAdherenceScore` |
| `rubric` | `RubricsScoreWithReference` |
| `data_creation` | `TestsetGenerator` synthetic dataset creation |
| `slow` | Tests making multiple live LLM/API calls |

## Environment

Copy `.env.example` to `.env` and fill in at minimum `OPENAI_API_KEY`. `RAG_API_BASE_URL` defaults to the Rahul Shetty Academy demo endpoint.

`uv` manages the virtualenv at `.venv/`. Python version is pinned in `.python-version` (3.12).
