"""Streamlit dashboard for RAG evaluation results.

Usage:
    uv run streamlit run scripts/view_results.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CSV_PATH = Path("reports/ragas_results.csv")

_METRIC_COLOURS = {
    "answer_relevancy": "#4CAF50",
    "context_precision": "#2196F3",
    "faithfulness": "#FF9800",
    "context_recall": "#9C27B0",
}


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Results file not found: {path}. Run `test_all_standard_metrics` first.")
        st.stop()
    return pd.read_csv(path)


def render_metric_cards(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    cols = st.columns(len(df.columns))
    for col, (metric, value) in zip(cols, latest.items()):
        colour = _METRIC_COLOURS.get(str(metric), "#607D8B")
        col.markdown(
            f"""<div style="border-top:4px solid {colour};padding:12px;border-radius:6px;
                            background:#fff;box-shadow:0 2px 6px rgba(0,0,0,.1)">
                <small style="color:#777;text-transform:uppercase">{str(metric).replace("_"," ")}</small>
                <div style="font-size:28px;font-weight:700;color:{colour}">{float(value):.1%}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
    st.title("RAG Evaluation Dashboard")

    df = load_results(_CSV_PATH)

    st.subheader("Latest Scores")
    render_metric_cards(df)

    st.subheader("Score History")
    st.line_chart(df[list(_METRIC_COLOURS)])

    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
