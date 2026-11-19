"""Checks for charts with more than one series.

Run from the project folder:  python test_charts.py

test_visualizer.py checks which chart *type* is chosen. This file checks
what is actually drawn when a result has two label columns (for example
location and category) or several measures (for example revenue,
expenses and surplus). No LLM or database is needed.
"""
from __future__ import annotations

import sys
from typing import Callable, List, Tuple

import pandas as pd

from src.agents.visualizer import build_chart

LOCATIONS = [
    "Mumbai Andheri", "Whitefield Auto Spa", "Bengaluru Central",
    "Kolkata Salt Lake", "Chennai Marina", "Delhi Connaught",
    "Pune Kothrud", "Hyderabad HiTech",
]
CATEGORIES = ["Salaries", "Rent", "Supplies", "Utilities", "Marketing", "Maintenance"]
MONTHS = [f"2024-{m:02d}" for m in range(1, 13)]


def _names(fig) -> List[str]:
    return [str(t.name) for t in fig.data]


def _expenses_by_location_and_category() -> pd.DataFrame:
    rows = [
        (i + 1, loc, cat, 1000.0 * (i + 1) + 100.0 * j)
        for j, cat in enumerate(CATEGORIES)
        for i, loc in enumerate(LOCATIONS)
    ]
    return pd.DataFrame(rows, columns=["id", "name", "category", "total_expenses"])


def _surplus_table() -> pd.DataFrame:
    revenue = [2068248.16, 2140193.08, 2131961.89, 2119331.92,
               2006284.79, 2069266.40, 1930372.32, 1729981.57]
    expenses = [1610627.51, 1629917.92, 1642432.42, 1649601.03,
                1629178.96, 1647351.55, 1625353.11, 1665927.30]
    return pd.DataFrame({
        "id": range(1, 9),
        "name": LOCATIONS,
        "total_revenue": revenue,
        "total_expenses": expenses,
        "operating_surplus": [r - e for r, e in zip(revenue, expenses)],
    })


def _per_employee_table() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [2, 7, 8],
        "name": ["Whitefield Auto Spa", "Delhi Connaught", "Kolkata Salt Lake"],
        "total_revenue": [6351132.66, 6413153.94, 6245128.67],
        "employee_count": [5, 7, 6],
        "revenue_per_employee": [1270226.53, 916164.85, 1040854.78],
    })


def case_two_label_columns() -> str:
    chart, fig = build_chart(_expenses_by_location_and_category())
    assert chart == "bar", chart
    assert len(fig.data) == 6, f"expected 6 series, got {len(fig.data)}"
    assert _names(fig) == CATEGORIES, _names(fig)
    assert all(len(list(t.x)) == 8 for t in fig.data)
    return "6 series, one per category, 8 locations each"


def case_similar_measures() -> str:
    chart, fig = build_chart(_surplus_table(), "SELECT ... ORDER BY operating_surplus DESC;")
    assert chart == "bar", chart
    assert _names(fig) == ["Total Revenue", "Total Expenses", "Operating Surplus"], _names(fig)
    return "3 series: revenue, expenses, surplus"


def case_mixed_scale_uses_sort_column() -> str:
    df = _per_employee_table()
    chart, fig = build_chart(df, "SELECT ... ORDER BY revenue_per_employee DESC")
    assert chart == "bar" and len(fig.data) == 1, (chart, len(fig.data))
    assert list(fig.data[0].y) == list(df["revenue_per_employee"])
    return "1 series: the column the query sorts by"


def case_mixed_scale_without_sql() -> str:
    df = _per_employee_table()
    chart, fig = build_chart(df)
    assert chart == "bar" and len(fig.data) == 1, (chart, len(fig.data))
    assert list(fig.data[0].y) == list(df["total_revenue"])
    return "1 series: the first measure"


def case_repeated_labels_become_table() -> str:
    df = pd.DataFrame({"name": ["A", "A", "B", "B"], "amount": [1.0, 2.0, 3.0, 4.0]})
    chart, fig = build_chart(df)
    assert chart == "table" and fig is None, chart
    return "table, not a misleading bar chart"


def case_too_many_series_become_table() -> str:
    rows = [(f"Location {i}", f"Service {j}", float(i * j + 1))
            for i in range(20) for j in range(12)]
    chart, fig = build_chart(pd.DataFrame(rows, columns=["name", "service", "revenue"]))
    assert chart == "table" and fig is None, chart
    return "table (20 x 12 is too crowded to chart)"


def case_axis_and_series_swap() -> str:
    regions = ["South", "West", "North", "East"]
    rows = [(r, f"Service {j}", float(10 * i + j + 1))
            for i, r in enumerate(regions) for j in range(12)]
    chart, fig = build_chart(pd.DataFrame(rows, columns=["region", "service", "revenue"]))
    assert chart == "bar", chart
    assert _names(fig) == regions, _names(fig)
    assert all(len(list(t.x)) == 12 for t in fig.data)
    return "12 services on the axis, 4 region series"


def case_line_with_two_measures() -> str:
    df = pd.DataFrame({
        "month": MONTHS,
        "monthly_revenue": [1.4e6 - 1e4 * i for i in range(12)],
        "monthly_expenses": [1.1e6 + 2e4 * i for i in range(12)],
    })
    chart, fig = build_chart(df)
    assert chart == "line", chart
    assert _names(fig) == ["Monthly Revenue", "Monthly Expenses"], _names(fig)
    return "2 lines: revenue and expenses"


def case_numeric_year_axis() -> str:
    df = pd.DataFrame({"year": [2023, 2024, 2025], "total_revenue": [4.1e6, 16.2e6, 17.0e6]})
    chart, fig = build_chart(df)
    assert chart == "line" and len(fig.data) == 1, (chart, len(fig.data))
    assert list(fig.data[0].x) == [2023, 2024, 2025]
    assert list(fig.data[0].y) == list(df["total_revenue"])
    return "year on the axis, revenue as the line"


def case_time_word_in_measure_name() -> str:
    df = pd.DataFrame({"region": ["South", "West", "North"],
                       "average_monthly_revenue": [680000.0, 330000.0, 178000.0]})
    chart, fig = build_chart(df)
    assert chart == "bar" and len(fig.data) == 1, (chart, len(fig.data))
    assert list(fig.data[0].x) == ["South", "West", "North"]
    return "bar chart by region, not a line chart"


def case_line_per_category_with_extra_measure() -> str:
    df = pd.DataFrame({
        "month": MONTHS * 2,
        "region": ["South"] * 12 + ["North"] * 12,
        "revenue": [float(i) for i in range(24)],
        "transaction_count": list(range(100, 124)),
    })
    chart, fig = build_chart(df)
    assert chart == "line", chart
    assert _names(fig) == ["South", "North"], _names(fig)
    assert list(fig.data[0].y) == [float(i) for i in range(12)]
    return "one line per region, using revenue"


def case_unaggregated_rows_become_table() -> str:
    df = pd.DataFrame({
        "id": [11, 12, 13, 14],
        "date": ["2024-03-01", "2024-03-01", "2024-03-02", "2024-03-02"],
        "total_revenue": [900.0, 450.0, 1200.0, 300.0],
    })
    chart, fig = build_chart(df)
    assert chart == "table" and fig is None, chart
    return "table, because dates repeat with nothing to tell rows apart"


CASES: List[Tuple[str, Callable[[], str]]] = [
    ("Expenses by location and category", case_two_label_columns),
    ("Revenue, expenses and surplus by location", case_similar_measures),
    ("Mixed scales, query sorts by one column", case_mixed_scale_uses_sort_column),
    ("Mixed scales, no SQL given", case_mixed_scale_without_sql),
    ("Repeated labels, nothing to tell them apart", case_repeated_labels_become_table),
    ("Too many series", case_too_many_series_become_table),
    ("Fewer regions than services", case_axis_and_series_swap),
    ("Monthly revenue and expenses", case_line_with_two_measures),
    ("Numeric year column", case_numeric_year_axis),
    ("Measure named after a time period", case_time_word_in_measure_name),
    ("Line per region with a second measure", case_line_per_category_with_extra_measure),
    ("Raw rows with repeated dates", case_unaggregated_rows_become_table),
]


def main() -> int:
    passed = 0
    for name, case in CASES:
        try:
            detail = case()
        except Exception as exc:  # noqa: BLE001 - report every failure, then exit non-zero
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            continue
        passed += 1
        print(f"[PASS] {name}: {detail}")
    print(f"\n{passed}/{len(CASES)} passed.")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
