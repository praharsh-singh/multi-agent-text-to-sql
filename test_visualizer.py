"""Unit tests for deterministic chart-type selection in the Visualizer.

Run with: python test_visualizer.py
"""
import pandas as pd
import plotly.graph_objects as go

from src.agents.visualizer import build_chart

months = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m-%d").tolist()

CASES = [
    ("Revenue by region",
     pd.DataFrame({"region": ["North", "South", "East", "West"],
                    "total_revenue": [6.2e6, 6.4e6, 24.3e6, 11.8e6]}), "bar"),
    ("Top services (id column must be ignored)",
     pd.DataFrame({"id": [7, 6, 5],
                    "name": ["Ceramic Coating", "Full Detailing", "Exterior Detailing"],
                    "total_revenue": [8.0e6, 2.1e6, 1.1e6]}), "bar"),
    ("Single top location (KPI)",
     pd.DataFrame({"id": [2], "name": ["Whitefield Auto Spa"],
                    "revenue_per_employee": [1270226.532]}), "metric"),
    ("Monthly revenue (ISO-date strings)",
     pd.DataFrame({"month": months, "revenue": range(100, 112)}), "line"),
    ("Yearly revenue (text year)",
     pd.DataFrame({"year": ["2023", "2024", "2025"], "revenue": [1, 2, 3]}), "line"),
    ("Monthly revenue per region (time + category)",
     pd.DataFrame({"month": months * 2, "region": ["South"] * 12 + ["West"] * 12,
                    "revenue": list(range(12)) + list(range(12, 24))}), "line"),
    ("Too many categories (40 rows)",
     pd.DataFrame({"name": [f"Item {i}" for i in range(40)],
                    "revenue": range(40)}), "table"),
    ("Text-only result",
     pd.DataFrame({"name": ["A", "B"], "city": ["Pune", "Delhi"]}), "table"),
    ("Empty result",
     pd.DataFrame({"region": [], "total_revenue": []}), "empty"),
    ("SUM over zero rows (single NULL)",
     pd.DataFrame({"total_revenue": [None]}), "empty"),
]


def main() -> None:
    failures = 0
    for label, df, expected in CASES:
        chart_type, fig = build_chart(df)
        has_fig = isinstance(fig, go.Figure)
        should_have_fig = expected not in ("empty", "table")
        ok = chart_type == expected and has_fig == should_have_fig
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: got '{chart_type}'"
              f"{'' if ok else f' (expected {expected})'}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed.")


if __name__ == "__main__":
    main()