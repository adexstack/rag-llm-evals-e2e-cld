# rag-llm-evals

End-to-end evaluation framework for a RAG (Retrieval-Augmented Generation) system using [Ragas](https://docs.ragas.io/) and OpenAI. Measures retrieval and generation quality across five metric families with a full pytest suite, structured logging, and an HTML/Streamlit dashboard.

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Tests](#running-the-tests)
- [Dashboards](#dashboards)
- [Project Structure](#project-structure)
- [Metrics Reference](#metrics-reference)
- [CI/CD Integration](#cicd-integration)
- [Extending the Framework](#extending-the-framework)

---

## Architecture

```
RAG API (external)
        │
        ▼
src/rag_evals/
  client.py      ← HTTP client with retry + timeout
  samples.py     ← builds Ragas sample objects from API responses or static data
  config.py      ← three-layer settings: env vars > config.json > built-in defaults
  reporting.py   ← writes results to reports/ragas_results.csv
  exceptions.py  ← typed exceptions (RagAPIError, ConfigurationError …)
        │
        ▼
tests/
  conftest.py               ← session-scoped LLM + embeddings fixtures
  test_retrieval.py         ← Context Precision, Context Recall
  test_generation.py        ← Faithfulness, Answer Relevancy, Answer Correctness
  test_standard_metrics.py  ← all four core metrics concurrently, writes CSV
  test_multi_turn.py        ← Topic Adherence (static + live conversation)
  test_rubric.py            ← Rubric-based scoring
  test_data_creation.py     ← Ragas TestsetGenerator smoke test
        │
        ▼
scripts/
  create_dashboard.py   ← generates reports/results_dashboard.html
  view_results.py       ← Streamlit live dashboard
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.12 |
| [uv](https://docs.astral.sh/uv/) | latest |
| OpenAI API key | required |

---

## Installation

```bash
# 1. Clone
git clone <repo-url>
cd rag-llm-evals-e2e-cld

# 2. Install dependencies into an isolated .venv
#    --extra test installs pytest, pytest-asyncio, pytest-html, rapidfuzz
uv sync --extra test

# 3. Configure secrets (see next section)
cp .env.example .env
```

---

## Configuration

Settings follow a three-layer priority stack: **env vars > `config.json` > built-in defaults**.

- **Secrets** (`OPENAI_API_KEY`, `RAGAS_APP_TOKEN`) are env-var only — never read from `config.json`.
- **Non-secret defaults** (models, timeout, thresholds) can be changed by editing `config.json` at the project root without touching environment files.
- **CI/CD** can override any value per-environment via env vars.

Copy `.env.example` to `.env` and fill in your values — **never commit `.env`** (it is gitignored).

```dotenv
# .env
OPENAI_API_KEY=sk-...          # required

# Optional — defaults shown
RAGAS_APP_TOKEN=               # only needed for ragas.app uploads
RAG_API_BASE_URL=https://rahulshettyacademy.com/rag-llm
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
API_TIMEOUT_SECONDS=30

# Set to false to skip ALL threshold assertions (useful for exploratory runs)
ASSERT_TEST_THRESHOLD=true

# Per-metric score thresholds — set to "" (empty string) to skip that metric's assertion
THRESHOLD_CONTEXT_PRECISION=0.8
THRESHOLD_CONTEXT_RECALL=0.7
THRESHOLD_FAITHFULNESS=0.8
THRESHOLD_ANSWER_RELEVANCY=0.8
THRESHOLD_TOPIC_ADHERENCE=0.8
THRESHOLD_RUBRIC_SCORE_MIN=2.0
```

The same keys are available in `config.json` (camelCase → snake_case, thresholds nested under `score_thresholds`). Example:

```json
{
  "assert_test_threshold": true,
  "score_thresholds": {
    "context_precision": "",
    "context_recall": 0.7
  }
}
```

Setting a threshold to `""` in `config.json` or as an env var disables the assertion for that metric without affecting others.

---

## Running the Tests

**Always use `python -m pytest`** — bare `pytest` skips editable-install resolution.

```bash
# Run the full suite
python -m pytest

# Run a specific module
python -m pytest tests/test_retrieval.py -v

# Run a single test
python -m pytest tests/test_generation.py::test_faithfulness -v

# Run by marker
python -m pytest -m retrieval
python -m pytest -m "not slow"

# Include tests that call the RAG API inside fixtures
python -m pytest --live

# Generate an HTML test report
python -m pytest --html=reports/test_report.html --self-contained-html
```

### Available markers

| Marker | Tests |
|--------|-------|
| `retrieval` | Context Precision, Context Recall |
| `generation` | Faithfulness, Answer Relevancy, Answer Correctness, all-metrics aggregate |
| `multi_turn` | Topic Adherence (static + live) |
| `rubric` | Rubric-based scoring |
| `data_creation` | TestsetGenerator smoke test |
| `slow` | Tests making multiple live API calls |

### Expected output

```
tests/test_retrieval.py::test_context_precision[...]       PASSED
tests/test_retrieval.py::test_context_recall[...]          PASSED
tests/test_generation.py::test_faithfulness[...]           PASSED
tests/test_standard_metrics.py::test_all_standard_metrics  PASSED   ← writes reports/ragas_results.csv
tests/test_multi_turn.py::test_topic_adherence_static      PASSED
tests/test_multi_turn.py::test_topic_adherence_live        PASSED   (requires --live)
tests/test_rubric.py::test_rubric_score                    PASSED
tests/test_data_creation.py::test_testset_generation       PASSED
```

---

## Dashboards

### Static HTML dashboard

Reads `reports/ragas_results.csv` (written by `test_all_standard_metrics`) and generates a self-contained HTML file.

```bash
uv run python scripts/create_dashboard.py

# Custom paths
uv run python scripts/create_dashboard.py \
  --input  reports/ragas_results.csv \
  --output reports/dashboard.html
```

Open `reports/results_dashboard.html` in any browser.

### Streamlit live dashboard

```bash
uv run streamlit run scripts/view_results.py
```

Displays score cards, a trend chart, and the raw CSV table at `http://localhost:8501`.

---

## Project Structure

```
rag-llm-evals-e2e-cld/
├── src/
│   └── rag_evals/
│       ├── __init__.py
│       ├── client.py        # RAG API HTTP client (retry, timeout, typed errors)
│       ├── config.py        # Settings: env vars > config.json > built-in defaults
│       ├── exceptions.py    # RagAPIError, RagAPITimeoutError, ConfigurationError
│       ├── reporting.py     # save_results() → CSV
│       └── samples.py       # sample-builder helpers (single-turn, multi-turn)
│
├── tests/
│   ├── conftest.py               # Session-scoped LLM + embeddings fixtures
│   ├── test_retrieval.py         # Context Precision, Context Recall
│   ├── test_generation.py        # Faithfulness, Answer Relevancy, Answer Correctness
│   ├── test_standard_metrics.py  # All four core metrics concurrently, writes CSV
│   ├── test_multi_turn.py        # Topic Adherence
│   ├── test_rubric.py            # Rubric scoring
│   └── test_data_creation.py     # Synthetic testset generation
│
├── testdata/
│   ├── precision_data.json
│   ├── recall_data.json
│   ├── faithfulness_data.json
│   ├── relevance_fact_data.json
│   ├── standard_metrics_data.json
│   ├── single_turn_sample.json
│   ├── multi_turn_static.json
│   ├── multi_turn_live.json
│   └── mock_docs.json
│
├── scripts/
│   ├── create_dashboard.py   # Generates HTML report from CSV
│   └── view_results.py       # Streamlit dashboard
│
├── reports/                  # Generated — gitignored
│
├── .github/
│   └── workflows/ci.yml      # GitHub Actions CI
│
├── config.json               # Non-secret defaults (overridable via env vars)
├── .env.example              # Safe config template
├── .gitignore
├── .python-version
├── conftest.py               # Root conftest — registers --live CLI flag
├── pyproject.toml
├── pytest.ini
└── uv.lock
```

---

## Metrics Reference

| Metric | Module | What it measures |
|--------|--------|-----------------|
| **Context Precision** | `test_retrieval.py` | Are retrieved chunks relevant to the question? |
| **Context Recall** | `test_retrieval.py` | Does the retrieved context cover the reference answer? |
| **Faithfulness** | `test_generation.py` | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | `test_generation.py` | Does the answer address the user's question? |
| **Answer Correctness** | `test_generation.py` | How factually close is the answer to the reference? |
| **Topic Adherence** | `test_multi_turn.py` | Does a multi-turn conversation stay on topic? |
| **Rubric Score** | `test_rubric.py` | LLM-judged score (1–5) against custom rubric criteria |

All score thresholds are configurable via `config.json` or environment variables (see [Configuration](#configuration)).

---

## CI/CD Integration

The repository ships a ready-to-use workflow at `.github/workflows/ci.yml` that runs on every push to `main`, every pull request, and on manual dispatch.

```yaml
# .github/workflows/ci.yml (actual workflow)
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Install dependencies
        run: uv sync --frozen --extra test

      - name: Run tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          RAGAS_APP_TOKEN: ${{ secrets.RAGAS_APP_TOKEN }}
        run: |
          uv run pytest \
            --html=reports/test_report.html \
            --self-contained-html \
            --junit-xml=reports/junit.xml

      - name: Generate results dashboard
        if: always()
        continue-on-error: true
        run: |
          uv run python scripts/create_dashboard.py \
            --input  reports/ragas_results.csv \
            --output reports/results_dashboard.html

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-reports-${{ github.run_number }}
          path: reports/
          retention-days: 30
```

Add `OPENAI_API_KEY` (and optionally `RAGAS_APP_TOKEN`) to your repository's **Settings → Secrets and variables → Actions**.

To relax thresholds for CI smoke tests without editing `config.json`, set env vars on the `Run tests` step:

```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  THRESHOLD_CONTEXT_RECALL: "0.5"
  ASSERT_TEST_THRESHOLD: "false"   # disable all assertions entirely
```

---

## Extending the Framework

### Add a new metric test

1. Create a test function in the appropriate `tests/test_*.py` file.
2. Use `build_single_turn_sample()` from `rag_evals.samples` for any test that hits the RAG API.
3. Add a threshold entry to `config.json` under `score_thresholds` and document the corresponding `THRESHOLD_*` env var in `.env.example`.

### Add new test data

Drop a JSON file into `testdata/` following the existing format (list of objects with at least a `"question"` key), then reference it in the `@pytest.mark.parametrize` decorator using `load_test_data("testdata/your_file.json")`.

### Point at a different RAG endpoint

```dotenv
RAG_API_BASE_URL=https://your-rag-backend.example.com/api
```

No code changes required.

### Switch the LLM or embedding model

```dotenv
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
```
