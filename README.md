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
  samples.py     ← builds Ragas SingleTurnSample from API response
  config.py      ← all settings from env vars (no hardcoded values)
  reporting.py   ← writes results to reports/ragas_results.csv
  exceptions.py  ← typed exceptions (RagAPIError, ConfigurationError …)
        │
        ▼
tests/
  conftest.py           ← session-scoped LLM + embeddings fixtures
  test_retrieval.py     ← Context Precision, Context Recall
  test_generation.py    ← Faithfulness, Answer Relevancy, all-metrics aggregate
  test_multi_turn.py    ← Topic Adherence (static + live conversation)
  test_rubric.py        ← Rubric-based scoring
  test_data_creation.py ← Ragas TestsetGenerator smoke test
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
uv sync

# 3. Configure secrets (see next section)
cp .env.example .env
```

---

## Configuration

All settings are read from environment variables. Copy `.env.example` to `.env` and fill in your values — **never commit `.env`** (it is gitignored).

```dotenv
# .env
OPENAI_API_KEY=sk-...          # required

# Optional — defaults shown
RAG_API_BASE_URL=https://rahulshettyacademy.com/rag-llm
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
API_TIMEOUT_SECONDS=30

# Score thresholds — lower these for CI smoke tests if needed
THRESHOLD_CONTEXT_PRECISION=0.8
THRESHOLD_CONTEXT_RECALL=0.7
THRESHOLD_FAITHFULNESS=0.8
THRESHOLD_ANSWER_RELEVANCY=0.8
THRESHOLD_TOPIC_ADHERENCE=0.8
THRESHOLD_RUBRIC_SCORE_MIN=2.0
```

> **Note:** `RAGAS_APP_TOKEN` is only required if you upload results to [ragas.app](https://ragas.app). Leave it blank otherwise.

---

## Running the Tests

```bash
# Run the full suite
pytest

# Run a specific module
pytest tests/test_retrieval.py -v

# Run a single test
pytest tests/test_generation.py::test_faithfulness -v

# Run by marker
pytest -m retrieval
pytest -m "not slow"

# Generate an HTML test report
pytest --html=reports/test_report.html --self-contained-html
```

### Available markers

| Marker | Tests |
|--------|-------|
| `retrieval` | Context Precision, Context Recall |
| `generation` | Faithfulness, Answer Relevancy, all-metrics aggregate |
| `multi_turn` | Topic Adherence (static + live) |
| `rubric` | Rubric-based scoring |
| `data_creation` | TestsetGenerator smoke test |
| `slow` | Tests making multiple live API calls |

### Expected output

```
tests/test_retrieval.py::test_context_precision[...] PASSED
tests/test_retrieval.py::test_context_recall[...]    PASSED
tests/test_generation.py::test_faithfulness[...]     PASSED
tests/test_generation.py::test_all_standard_metrics  PASSED   ← writes reports/ragas_results.csv
tests/test_multi_turn.py::test_topic_adherence_static PASSED
tests/test_multi_turn.py::test_topic_adherence_live  PASSED
tests/test_rubric.py::test_rubric_score              PASSED
tests/test_data_creation.py::test_testset_generation PASSED
```

---

## Dashboards

### Static HTML dashboard

Reads `reports/ragas_results.csv` (written by `test_all_standard_metrics`) and generates a self-contained HTML file.

```bash
.venv/bin/python scripts/create_dashboard.py

# Custom paths
.venv/bin/python scripts/create_dashboard.py \
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
│       ├── config.py        # Settings dataclass loaded from env vars
│       ├── exceptions.py    # RagAPIError, RagAPITimeoutError, ConfigurationError
│       ├── reporting.py     # save_results() → CSV
│       └── samples.py       # build_single_turn_sample() helper
│
├── tests/
│   ├── conftest.py           # Session-scoped LLM + embeddings fixtures
│   ├── test_retrieval.py     # Context Precision, Context Recall
│   ├── test_generation.py    # Faithfulness, Answer Relevancy, all-metrics
│   ├── test_multi_turn.py    # Topic Adherence
│   ├── test_rubric.py        # Rubric scoring
│   └── test_data_creation.py # Synthetic testset generation
│
├── testdata/
│   ├── precision_data.json
│   ├── recall_data.json
│   ├── faithfulness_data.json
│   └── relevance_fact.json
│
├── scripts/
│   ├── create_dashboard.py   # Generates HTML report from CSV
│   └── view_results.py       # Streamlit dashboard
│
├── reports/                  # Generated — gitignored
│
├── .env.example              # Safe config template
├── .gitignore
├── .python-version
├── conftest.py               # Root conftest (sys.path bootstrap only)
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

All score thresholds are configurable via environment variables (see [Configuration](#configuration)).

---

## CI/CD Integration

### GitHub Actions example

```yaml
# .github/workflows/rag-evals.yml
name: RAG Evaluations

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 6 * * 1"   # weekly Monday 06:00 UTC

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - name: Install dependencies
        run: uv sync

      - name: Run evaluations
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          THRESHOLD_CONTEXT_RECALL: "0.5"   # relaxed for CI
        run: |
          uv run pytest \
            --html=reports/test_report.html \
            --self-contained-html \
            -m "not slow"

      - name: Generate dashboard
        run: uv run python scripts/create_dashboard.py

      - name: Upload reports
        uses: actions/upload-artifact@v4
        with:
          name: rag-eval-reports
          path: reports/
```

> Add `OPENAI_API_KEY` to your repository's **Settings → Secrets and variables → Actions**.

---

## Extending the Framework

### Add a new metric test

1. Create a test function in the appropriate `tests/test_*.py` file.
2. Use `build_single_turn_sample()` from `rag_evals.samples` for any test that hits the RAG API.
3. Add a threshold to `.env.example` and `config.py → _default_thresholds()`.

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
