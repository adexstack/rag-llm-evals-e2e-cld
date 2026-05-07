"""Generate a self-contained HTML dashboard from ragas_results.csv.

Usage:
    uv run python scripts/create_dashboard.py
    uv run python scripts/create_dashboard.py --input reports/ragas_results.csv --output reports/dashboard.html
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_METRIC_COLORS = {
    "answer_relevancy": "#4CAF50",
    "context_precision": "#2196F3",
    "faithfulness": "#FF9800",
    "context_recall": "#9C27B0",
}


def _metric_card(name: str, value: float) -> str:
    colour = _METRIC_COLORS.get(name, "#607D8B")
    pct = f"{value * 100:.1f}%"
    label = name.replace("_", " ").title()
    return f"""
    <div class="card" style="border-top: 4px solid {colour};">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{colour};">{pct}</div>
        <div class="metric-bar">
            <div class="bar-fill" style="width:{pct}; background:{colour};"></div>
        </div>
    </div>"""


def _table_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v:.4f}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    return "\n".join(rows)


def build_html(df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    cards = "".join(_metric_card(col, latest[col]) for col in df.columns)
    headers = "".join(f"<th>{c.replace('_', ' ').title()}</th>" for c in df.columns)
    rows = _table_rows(df)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Evaluation Dashboard</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
  h1 {{ text-align: center; color: #333; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
  .card {{ background: #fff; border-radius: 8px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,.12); }}
  .metric-label {{ font-size: 13px; color: #777; text-transform: uppercase; letter-spacing: .5px; }}
  .metric-value {{ font-size: 36px; font-weight: 700; margin: 8px 0; }}
  .metric-bar {{ background: #eee; border-radius: 4px; height: 6px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px;
           overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,.12); }}
  th {{ background: #333; color: #fff; padding: 12px 16px; text-align: left; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #eee; }}
  tr:last-child td {{ border-bottom: none; }}
</style>
</head>
<body>
<h1>RAG Evaluation Dashboard</h1>
<div class="grid">{cards}</div>
<h2 style="color:#333;">All Results</h2>
<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate RAG evaluation HTML dashboard")
    parser.add_argument("--input", default="reports/ragas_results.csv")
    parser.add_argument("--output", default="reports/results_dashboard.html")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    df = pd.read_csv(input_path)
    if df.empty:
        logger.error("CSV file is empty: %s", input_path)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(df)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
