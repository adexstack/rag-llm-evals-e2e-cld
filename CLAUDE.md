# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (creates .venv automatically)
# --extra test installs pytest/pytest-asyncio/pytest-html/rapidfuzz
uv sync --extra test

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

# Run only tests that hit the live RAG API via fixtures
python -m pytest --live

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
1. `testdata/*.json` → `load_test_data()` / `load_sample()` / `load_documents()` → parametrize decorator or fixture
2. Indirect fixture (`get_precision_sample`, etc.) → `build_single_turn_sample()` → calls live RAG API via `RagClient`
3. `SingleTurnSample` → Ragas metric `.ascore()` → score compared against threshold from `Settings.threshold_for()`

**`src/rag_evals/` modules:**
- `config.py` — Frozen `Settings` dataclass with a three-layer priority stack: **env vars > `config.json` > built-in fallbacks**. Singleton via `lru_cache`. `OPENAI_API_KEY` is required (env-var only). All non-secret settings can be changed in `config.json` at the project root; CI/CD can override per-env via `THRESHOLD_*` / `ASSERT_TEST_THRESHOLD` env vars. Setting a threshold to `""` (empty string) in `config.json` or as an env var skips the assertion for that metric. `assert_test_threshold: false` disables all threshold assertions at once. `Settings.threshold_for(metric)` returns the resolved threshold or `None` when assertions are disabled.
- `client.py` — `RagClient` wraps the `/ask` endpoint with retry (3×, backoff 0.5, on 5xx), 30s timeout. Returns `{"answer": str, "retrieved_docs": [{"page_content": str}]}`.
- `samples.py` — Collection of helpers:
  - `load_test_data(path)` — loads a JSON array for `@pytest.mark.parametrize`
  - `load_sample(path)` — loads a single JSON object
  - `load_documents(path)` — loads LangChain `Document` objects + expected sample count (for `TestsetGenerator`)
  - `build_single_turn_sample()` — calls `RagClient.ask()` and builds a `SingleTurnSample`
  - `build_single_turn_from_data()` — builds a `SingleTurnSample` from a pre-fetched data dict (no API call)
  - `build_multi_turn_sample_static()` — builds a `MultiTurnSample` from static conversation JSON
  - `build_multi_turn_sample_live()` — builds a `MultiTurnSample` by calling the RAG API for each question
  - All path helpers resolve paths relative to the project root regardless of CWD.
- `reporting.py` — `save_results()` writes a single-row CSV to `reports/ragas_results.csv`.
- `exceptions.py` — `RagAPIError`, `RagAPITimeoutError`, `ConfigurationError`.

**`--live` CLI flag:** The root `conftest.py` registers a `--live` option. Tests that call the RAG API inside fixtures (e.g. `get_topic_data_live`) are skipped unless `--live` is passed. Tests that call the API directly (parametrized with `indirect=True`) always make live calls.

**Ragas metric return types:** `ascore()` returns a `MetricResult` object, not a plain `float`. It supports `%`-style formatting and `float()` conversion, but **not** f-string `:.4f` format specs. Always wrap in `float()` when formatting in assert messages: `f"score={float(score):.4f}"`.

**`TestsetGenerator` import:** Must be imported inside the test function body in `test_data_creation.py`. Importing at module scope causes a `PytestCollectionWarning` because the class name starts with `Test`.

## Test Markers

| Marker | What it covers |
|---|---|
| `retrieval` | `ContextPrecisionWithoutReference`, `ContextRecall` |
| `generation` | `Faithfulness`, `AnswerRelevancy`, `AnswerCorrectness`, all-metrics aggregate |
| `multi_turn` | `TopicAdherenceScore` |
| `rubric` | `RubricsScoreWithReference` |
| `data_creation` | `TestsetGenerator` synthetic dataset creation |
| `slow` | Tests making multiple live LLM/API calls |

## Environment

Copy `.env.example` to `.env` and fill in at minimum `OPENAI_API_KEY`. `RAG_API_BASE_URL` defaults to the Rahul Shetty Academy demo endpoint.

Non-secret defaults (models, timeout, thresholds) can also be changed in `config.json` at the project root without touching env vars.

`uv` manages the virtualenv at `.venv/`. Python version is pinned in `.python-version` (3.12).
