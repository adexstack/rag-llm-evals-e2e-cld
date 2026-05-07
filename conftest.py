# Root conftest.py — kept minimal.
# All fixtures live in tests/conftest.py.
# This file exists only to ensure src/ is on sys.path when running pytest
# from the project root with `uv run pytest`.


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that make live calls to the RAG API inside fixtures.",
    )
