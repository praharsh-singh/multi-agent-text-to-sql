"""Deterministic chart-type selection and Plotly figure building for
Text-to-SQL query results.

This module inspects a pandas DataFrame returned by the Critic/executor
and picks the most sensible chart type for it, without ever asking an
LLM to decide. It is intentionally conservative: when the shape of the
result doesn't map cleanly onto a chart, it falls back to a table (or a
plain empty state) rather than guessing.

A chart can carry more than one series in two situations:

* The result has two label columns (for example location and expense
  category). The second one becomes the series, so each category gets
  its own colour.
* The result has several measures on a similar scale (for example
  revenue, expenses and surplus). Each measure becomes a series.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from src.agents.state import AnalysisState
from src.logger import get_logger

logger = get_logger(__name__)

_MAX_BAR_CATEGORIES = 20
_MAX_SERIES = 6
_MAX_MEASURE_SERIES = 4
_COMPARABLE_SCALE_RATIO = 25.0
_TIME_COLUMN_HINTS = ("date", "month", "year", "week", "day", "quarter", "time")
_TIME_UNIT_NAMES = ("year", "month", "quarter", "week", "day")
_ID_COLUMN_SUFFIXES = ("_id",)
_ID_COLUMN_NAMES = ("id",)
_GENERIC_AXIS_NAMES = ("name",)
_ORDER_BY_PATTERN = re.compile(r"ORDER\s+BY\s+([A-Za-z_][\w.]*)", re.IGNORECASE)


def _is_id_column(column_name: str) -> bool:
    """Check whether a column looks like a primary/foreign key rather
    than a real measure or label.

    Args:
        column_name: The column name to check.

    Returns:
        True if the column name matches "id" or ends in "_id".
    """
    name = column_name.lower()
    return name in _ID_COLUMN_NAMES or name.endswith(_ID_COLUMN_SUFFIXES)


def _is_time_like_column(column_name: str) -> bool:
    """Check whether a column name suggests a time axis (date, month,
    year, etc.), used to prefer a line chart over a bar chart.

    Args:
        column_name: The column name to check.

    Returns:
        True if the column name contains a recognized time-related word.
    """
    name = column_name.lower()
    return any(hint in name for hint in _TIME_COLUMN_HINTS)


def _is_time_unit_name(column_name: str) -> bool:
    """Check whether a column is named after a time unit itself.

    A numeric column called ``year`` is a time axis; a numeric column
    called ``monthly_revenue`` is a measure that merely mentions time.

    Args:
        column_name: The column name to check.

    Returns:
        True for names such as "year", "month" or "order_year".
    """
    name = column_name.lower()
    return name in _TIME_UNIT_NAMES or name.endswith(
        tuple(f"_{unit}" for unit in _TIME_UNIT_NAMES)
    )


def _title(column_name: str) -> str:
    """Turn a column name like ``total_revenue`` into ``Total Revenue``.

    Args:
        column_name: The raw column name.

    Returns:
        A title-case label for an axis or legend.
    """
    return column_name.replace("_", " ").title()


def _axis_title(column_name: str) -> Optional[str]:
    """Return an axis title, or None when the name says nothing useful.

    Args:
        column_name: The column shown on the axis.

    Returns:
        A title, or None for generic names such as "name".
    """
    if column_name.lower() in _GENERIC_AXIS_NAMES:
        return None
    return _title(column_name)


def _usable_columns(df: pd.DataFrame) -> List[str]:
    """List the columns worth charting, leaving out key columns.

    Args:
        df: The query result.

    Returns:
        Column names that are not ids.
    """
    return [c for c in df.columns if not _is_id_column(c)]


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """List the numeric, non-id columns of a result.

    Args:
        df: The query result.

    Returns:
        Names of numeric columns that are not ids.
    """
    return [
        c for c in _usable_columns(df) if pd.api.types.is_numeric_dtype(df[c])
    ]


def _time_axis(df: pd.DataFrame) -> Optional[str]:
    """Find the column to use as a time axis, if the result has one.

    Text or date columns with a time-like name come first. A numeric
    column only counts when it is named after a time unit (``year``),
    so a measure such as ``monthly_revenue`` is never used as the axis.

    Args:
        df: The query result.

    Returns:
        The time-axis column name, or None.
    """
    numeric = _numeric_columns(df)
    time_like = [c for c in _usable_columns(df) if _is_time_like_column(c)]
    for column in time_like:
        if column not in numeric:
            return column
    for column in time_like:
        if _is_time_unit_name(column):
            return column
    return None


def _order_by_column(sql: Optional[str]) -> Optional[str]:
    """Read the column the query sorts by, as a hint for the main measure.

    Args:
        sql: The SQL that produced the result, if known.

    Returns:
        The bare column name of the last ORDER BY, lower-cased, or None.
    """
    if not sql:
        return None
    matches = _ORDER_BY_PATTERN.findall(sql)
    if not matches:
        return None
    return matches[-1].split(".")[-1].lower()


def _primary_measure(measures: List[str], sql: Optional[str]) -> str:
    """Pick the one measure to plot when only one can be shown.

    The column the query sorts by is the one the question ranks on, so
    it wins. Otherwise the first measure is used.

    Args:
        measures: Candidate measure columns (at least one).
        sql: The SQL that produced the result, if known.

    Returns:
        The chosen measure column.
    """
    ordered_by = _order_by_column(sql)
    for column in measures:
        if column.lower() == ordered_by:
            return column
    return measures[0]


def _comparable_measures(df: pd.DataFrame, measures: List[str]) -> bool:
    """Check whether several measures can share one axis.

    Revenue and expenses can; revenue and an employee count cannot,
    because the small one would vanish next to the large one.

    Args:
        df: The query result.
        measures: The candidate measure columns.

    Returns:
        True if there are 2 to 4 measures whose largest absolute values
        are within a factor of 25 of each other.
    """
    if not 2 <= len(measures) <= _MAX_MEASURE_SERIES:
        return False
    peaks = [float(df[m].abs().max()) for m in measures]
    if any(pd.isna(p) or p <= 0 for p in peaks):
        return False
    return max(peaks) / min(peaks) <= _COMPARABLE_SCALE_RATIO


def _bar_layout(
    df: pd.DataFrame, categories: List[str]
) -> Optional[Tuple[str, Optional[str]]]:
    """Decide which label column is the x-axis and which is the series.

    Args:
        df: The query result.
        categories: Non-numeric, non-id columns, in result order.

    Returns:
        ``(x_column, series_column)``, where ``series_column`` is None
        for a plain bar chart. None if the result can't be drawn as a
        bar chart without misleading (too many bars, or repeated labels
        with nothing to tell them apart).
    """
    if not categories:
        return None
    first = categories[0]
    if not df[first].duplicated().any():
        if df[first].nunique() <= _MAX_BAR_CATEGORIES:
            return first, None
        return None
    if len(categories) < 2:
        return None
    second = categories[1]
    for x_column, series_column in ((first, second), (second, first)):
        if (
            df[x_column].nunique() <= _MAX_BAR_CATEGORIES
            and df[series_column].nunique() <= _MAX_SERIES
            and not df.duplicated([x_column, series_column]).any()
        ):
            return x_column, series_column
    return None


def _line_is_readable(
    df: pd.DataFrame, time_column: str, categories: List[str]
) -> bool:
    """Check that each point on a line chart would be unambiguous.

    Args:
        df: The query result.
        time_column: The column used as the time axis.
        categories: Non-numeric, non-id columns other than the time axis.

    Returns:
        True if every time value appears once, or once per value of the
        first category column (which then becomes one line each).
    """
    if not df[time_column].duplicated().any():
        return True
    if not categories:
        return False
    return not df.duplicated([time_column, categories[0]]).any()


def choose_chart_type(df: pd.DataFrame) -> str:
    """Decide which chart type best fits a query result, using only the
    result's shape and column names/types — never the question text or
    an LLM call.

    Rules, in order:
        1. Empty result, or a single row whose only numeric value is
           null -> "empty".
        2. A single row with at least one real numeric value -> "metric".
        3. A time axis and at least one measure -> "line", provided each
           time value appears once (or once per category).
        4. A label column (ignoring id columns) with a reasonable number
           of distinct values and a numeric measure -> "bar". Repeated
           labels are allowed only when a second label column tells the
           rows apart; it becomes the series.
        5. Anything else (too many categories, no numeric column,
           non-chartable shape) -> "table".

    Args:
        df: The query result to classify.

    Returns:
        One of "empty", "metric", "line", "bar", "table".
    """
    if df is None or df.empty:
        return "empty"

    numeric_columns = _numeric_columns(df)

    if len(df) == 1:
        if numeric_columns and df[numeric_columns].iloc[0].notna().any():
            return "metric"
        return "empty"

    if not numeric_columns:
        return "table"

    time_column = _time_axis(df)
    measures = [c for c in numeric_columns if c != time_column]
    if not measures:
        return "table"
    category_columns = [
        c for c in _usable_columns(df)
        if c not in numeric_columns and c != time_column
    ]
    if time_column:
        if _line_is_readable(df, time_column, category_columns):
            return "line"
        return "table"

    if _bar_layout(df, category_columns) is not None:
        return "bar"

    return "table"


def _build_metric(df: pd.DataFrame) -> go.Figure:
    """Build a single big-number indicator figure for a one-row result.

    Args:
        df: A one-row DataFrame with at least one numeric column.

    Returns:
        A Plotly indicator Figure showing the first numeric value.
    """
    numeric_columns = _numeric_columns(df)
    value_column = numeric_columns[0]
    value = df[value_column].iloc[0]
    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=float(value),
            title={"text": _title(value_column)},
        )
    )
    return fig


def _build_line(df: pd.DataFrame, sql: Optional[str] = None) -> go.Figure:
    """Build a line chart with one line per category or per measure.

    Args:
        df: The query result containing a time axis and at least one
            numeric measure.
        sql: The SQL that produced the result, used to pick the main
            measure when there are several.

    Returns:
        A Plotly line Figure.
    """
    numeric_columns = _numeric_columns(df)
    x_column = _time_axis(df)
    measures = [c for c in numeric_columns if c != x_column]
    category_columns = [
        c for c in _usable_columns(df)
        if c not in numeric_columns and c != x_column
    ]
    y_title: Optional[str]

    fig = go.Figure()
    if category_columns:
        y_column = _primary_measure(measures, sql)
        group_column = category_columns[0]
        for group_value, group_df in df.groupby(group_column, sort=False):
            fig.add_trace(
                go.Scatter(
                    x=group_df[x_column],
                    y=group_df[y_column],
                    mode="lines+markers",
                    name=str(group_value),
                )
            )
        y_title = _title(y_column)
    elif _comparable_measures(df, measures):
        for measure in measures:
            fig.add_trace(
                go.Scatter(
                    x=df[x_column],
                    y=df[measure],
                    mode="lines+markers",
                    name=_title(measure),
                )
            )
        y_title = None
    else:
        y_column = _primary_measure(measures, sql)
        fig.add_trace(
            go.Scatter(x=df[x_column], y=df[y_column], mode="lines+markers")
        )
        y_title = _title(y_column)

    fig.update_layout(xaxis_title=_title(x_column), yaxis_title=y_title)
    return fig


def _build_bar(df: pd.DataFrame, sql: Optional[str] = None) -> go.Figure:
    """Build a bar chart, grouped when there is a series to show.

    Args:
        df: The query result containing a label column (ignoring id
            columns) and at least one numeric measure.
        sql: The SQL that produced the result, used to pick the main
            measure when there are several.

    Returns:
        A Plotly bar Figure.
    """
    measures = _numeric_columns(df)
    category_columns = [c for c in _usable_columns(df) if c not in measures]
    layout = _bar_layout(df, category_columns)
    if layout is None:
        raise ValueError("Result is not suitable for a bar chart.")
    x_column, series_column = layout
    y_title: Optional[str]

    fig = go.Figure()
    if series_column is not None:
        y_column = _primary_measure(measures, sql)
        for series_value in df[series_column].drop_duplicates():
            series_df = df[df[series_column] == series_value]
            fig.add_trace(
                go.Bar(
                    x=series_df[x_column],
                    y=series_df[y_column],
                    name=str(series_value),
                )
            )
        y_title = _title(y_column)
    elif _comparable_measures(df, measures):
        for measure in measures:
            fig.add_trace(
                go.Bar(x=df[x_column], y=df[measure], name=_title(measure))
            )
        y_title = None
    else:
        y_column = _primary_measure(measures, sql)
        fig.add_trace(go.Bar(x=df[x_column], y=df[y_column]))
        y_title = _title(y_column)

    fig.update_layout(
        barmode="group",
        xaxis_title=_axis_title(x_column),
        yaxis_title=y_title,
    )
    return fig


def build_chart(
    df: pd.DataFrame, sql: Optional[str] = None
) -> Tuple[str, Optional[go.Figure]]:
    """Classify a query result and build the matching Plotly figure.

    Args:
        df: The query result to visualize.
        sql: The SQL that produced the result. Optional; only used to
            pick the main measure when the result has several.

    Returns:
        A tuple of (chart_type, figure). figure is None for "empty" and
        "table" chart types, since those have no chart to render.
    """
    chart_type = choose_chart_type(df)

    if chart_type == "metric":
        return chart_type, _build_metric(df)
    if chart_type == "line":
        return chart_type, _build_line(df, sql)
    if chart_type == "bar":
        return chart_type, _build_bar(df, sql)

    return chart_type, None


def visualize(state: AnalysisState) -> AnalysisState:
    """LangGraph node: attach a chart type and figure to the state based
    on the Critic's execution result, if any.

    Args:
        state: The current graph state, after the critic node has run.

    Returns:
        Updated state with ``chart_type`` and ``figure`` set. If there
        is no result to visualize (an unanswerable or failed run), both
        are set to safe defaults rather than raising.
    """
    result = state.get("execution_result")

    if result is None:
        state["chart_type"] = "empty"
        state["figure"] = None
        return state

    try:
        chart_type, fig = build_chart(result, state.get("sql_query"))
    except Exception as exc:  # noqa: BLE001 - visualization must never crash the pipeline
        logger.warning("Visualizer: failed to build chart (%s). Falling back to table.", exc)
        chart_type, fig = "table", None

    state["chart_type"] = chart_type
    state["figure"] = fig
    logger.info("Visualizer: chart_type='%s'.", chart_type)
    return state
