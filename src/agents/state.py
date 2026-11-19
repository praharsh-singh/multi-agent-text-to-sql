"""Shared state object for the Text-to-SQL agent pipeline.

A single ``AnalysisState`` dict flows through every agent. Each agent reads
the fields it needs and writes the fields it produces, leaving the rest
untouched. This is the state schema LangGraph will use in Phase 3.
"""

from typing import Any, Optional, TypedDict

import pandas as pd
import plotly.graph_objects as go


class AnalysisState(TypedDict, total=False):
    """Carries all data through the Planner -> Programmer -> Critic -> Visualizer pipeline.

    ``total=False`` means every key is optional, so agents can populate the
    state incrementally as the pipeline progresses.

    Attributes:
        question:         User's natural-language question.
        schema_context:   Schema description string from SchemaInspector.
        query_plan:       Planner's structured plan (tables, columns, approach).
        sql_query:        Programmer's generated SQL string.
        execution_result: Critic's successful query result as a DataFrame.
        error_message:    Critic's error text if execution failed (else "").
        retry_count:      Number of Programmer retries attempted so far.
        row_count:        Number of rows returned by the successful query.
        final_status:     "success" | "failed" | "unanswerable" once the
                           pipeline terminates.
        chart_type:       Visualizer's chosen chart type ("metric", "line",
                           "bar", "table", or "empty").
        figure:           Visualizer's Plotly Figure for the result, or
                           None when chart_type is "table" or "empty".
    """

    question: str
    schema_context: str
    query_plan: str
    sql_query: str
    execution_result: Optional[pd.DataFrame]
    error_message: str
    retry_count: int
    row_count: int
    final_status: str
    chart_type: str
    figure: Optional[go.Figure]