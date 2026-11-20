"""Streamlit front end for the multi-agent Text-to-SQL data analyst.

Run from the project root with:
    streamlit run app.py

The UI is a thin layer over the LangGraph pipeline in ``src/graph.py``. It
streams each agent step while it works, then shows the answer as a chart,
single number or table, with the plan and SQL available for inspection.
Each question is answered independently; there is no follow-up memory.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st

from src.config import (
    DATABASE_PATH,
    LLM_PROVIDER,
    MAX_SQL_RETRIES,
    OLLAMA_MODEL,
    OPENAI_MODEL,
)
from src.graph import build_graph
from src.logger import get_logger
from src.question_parts import split_question
from src.schema_inspector import SchemaInspector

logger = get_logger(__name__)

# (question, Material icon hinting at the kind of answer it produces)
EXAMPLE_QUESTIONS: list[tuple[str, str]] = [
    ("What is the total revenue by region?", ":material/bar_chart:"),
    ("Which 5 services earned the most revenue?", ":material/leaderboard:"),
    ("What is the average revenue per employee at Whitefield Auto Spa?", ":material/numbers:"),
    ("What was the monthly revenue in 2024?", ":material/show_chart:"),
    ("What is the average customer satisfaction score?", ":material/help:"),
]

PIPELINE_STEPS: list[tuple[str, str]] = [
    ("Planner", "Finds the tables and columns the question needs, or says the data isn't there."),
    ("Programmer", "Writes a read-only SQL query from that plan, and fixes it if the Critic objects."),
    ("Critic", "Checks the query, runs it, and sends it back if something's wrong."),
    ("Visualizer", "Picks a chart, a single number or a table for the result."),
]

# Chart colours, used when .streamlit/config.toml doesn't supply any.
# Keep in step with chartCategoricalColors in that file.
_DEFAULT_CHART_COLORS: tuple[str, ...] = (
    "#0E6F7A",
    "#D9922E",
    "#4F6FA8",
    "#B5485A",
    "#5E8C4A",
    "#7A5C99",
)

_SCHEMA_TABLE_LINE = re.compile(r"^\s*Table:\s*([A-Za-z_]\w*)")
_SCHEMA_COLUMN_LINE = re.compile(r"^\s*-\s+([A-Za-z_]\w*)\s*\(")

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/query_stats:"

_TRACE_LABELS: dict[str, str] = {
    "planner": "Planner",
    "programmer": "Programmer",
    "critic": "Critic",
    "visualize": "Visualizer",
}
_CHART_LABELS: dict[str, str] = {
    "bar": "a bar chart",
    "line": "a line chart",
    "metric": "a single number",
    "table": "a table",
    "empty": "an empty-result notice",
}
_TIME_HINTS = ("date", "month", "year", "week", "day", "quarter", "time")
_UNANSWERABLE_PREFIX = re.compile(r"^\s*UNANSWERABLE\s*:\s*", re.IGNORECASE)
_READ_AS_LABEL = "READ AS:"
_ALSO_ASKED_LABEL = "ALSO ASKED:"
_NOTHING_LEFT_OUT = re.compile(
    r"^(?:(?:none|nothing|n/?a|not applicable|no other|no part)\b|[-–—.\s]*$)", re.IGNORECASE
)
_ORDER_BY_PATTERN = re.compile(r"ORDER\s+BY\s+([A-Za-z_][\w.]*)", re.IGNORECASE)
_MAX_METRICS = 6


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_graph() -> Any:
    """Build and cache the compiled LangGraph workflow.

    Returns:
        The compiled graph, shared across all sessions of this app.
    """
    return build_graph()


@st.cache_data(show_spinner=False)
def get_schema_context() -> str:
    """Build and cache the schema description fed to the Planner.

    Returns:
        The full schema description, including sample rows.
    """
    return SchemaInspector().get_schema_description(include_samples=True)


@st.cache_data(show_spinner=False)
def get_schema_overview() -> tuple[list[str], str]:
    """Build and cache the schema summary shown in the sidebar.

    Returns:
        A tuple of (table names, schema description without sample rows).
    """
    inspector = SchemaInspector()
    return list(inspector.get_table_names()), inspector.get_schema_description(
        include_samples=False
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _model_label() -> str:
    """Describe the configured LLM for display.

    Returns:
        A short human-readable model name.
    """
    if LLM_PROVIDER == "openai":
        return f"OpenAI {OPENAI_MODEL}"
    return f"Ollama {OLLAMA_MODEL}"


def _preflight_problems() -> list[str]:
    """Check the things that must be in place before any question can run.

    Returns:
        A list of human-readable problems; empty if everything is ready.
    """
    problems: list[str] = []
    if not DATABASE_PATH.exists():
        problems.append(
            f"The database file wasn't found at `{DATABASE_PATH}`. Run "
            "`python data/seed_database.py` from the project root, then "
            "refresh this page."
        )
    if LLM_PROVIDER == "openai" and not _load_api_key():
        problems.append(
            "`OPENAI_API_KEY` isn't set. On your computer: stop the app, run "
            "`source ~/.zshrc`, then start it again with `streamlit run app.py`. "
            "On Streamlit Community Cloud: open the app's **Settings → Secrets** "
            "and add the line `OPENAI_API_KEY = \"sk-...\"`."
        )
    return problems


def _load_api_key() -> bool:
    """Makes sure the OpenAI key is in the environment, where the model client looks.

    On a computer the key comes from the shell. On Streamlit Community Cloud
    it is stored as an app secret, which isn't always copied into the
    environment before this check runs, so it is copied here. Secrets are
    only read when the shell has no key, so a local run never touches them.

    Returns:
        True if the key is available.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return True
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:  # noqa: BLE001 - no secrets file, or it can't be parsed
        key = None
    if isinstance(key, str) and key.strip():
        os.environ["OPENAI_API_KEY"] = key.strip()
        return True
    return False


def _plural(count: int, noun: str) -> str:
    """Format a count with a correctly pluralised noun.

    Args:
        count: The number of things.
        noun: The singular noun.

    Returns:
        Text such as "1 row" or "4 rows".
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _shorten(text: str, limit: int = 160) -> str:
    """Trim an error message to its first line and a readable length.

    Args:
        text: The text to shorten.
        limit: Maximum number of characters to keep.

    Returns:
        The shortened text.
    """
    stripped = str(text).strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    return first_line if len(first_line) <= limit else first_line[: limit - 1] + "…"


def _is_id_column(column_name: str) -> bool:
    """Check whether a column is a key rather than a measure.

    Args:
        column_name: The column name to check.

    Returns:
        True if the name is "id" or ends in "_id".
    """
    name = str(column_name).lower()
    return name == "id" or name.endswith("_id")


def _is_time_like(column_name: str) -> bool:
    """Check whether a column name suggests a date or period.

    Args:
        column_name: The column name to check.

    Returns:
        True if the name contains a time-related word.
    """
    name = str(column_name).lower()
    return any(hint in name for hint in _TIME_HINTS)


def _format_number(value: Any) -> str:
    """Format a number with thousands separators.

    Args:
        value: The number to format.

    Returns:
        The formatted number, or the value as text if it isn't numeric.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.2f}"


def _pretty(column_name: str) -> str:
    """Turn a column name like ``total_revenue`` into ``Total revenue``.

    Args:
        column_name: The raw column name.

    Returns:
        A sentence-case label.
    """
    return str(column_name).replace("_", " ").strip().capitalize()


def _split_plan(plan: str) -> tuple[str, str, str]:
    """Separate the Planner's two labelled notes from the numbered plan.

    The Planner may add a ``READ AS:`` line (how it interpreted the
    question) and an ``ALSO ASKED:`` line (a part of the question the plan
    leaves out). Both are shown to the user outside the plan.

    Args:
        plan: The Planner's raw output.

    Returns:
        A tuple of (read-as text, also-asked text, remaining plan). The
        first two are empty strings when the Planner didn't write them.
    """
    read_as = ""
    also_asked = ""
    body: list[str] = []
    for line in (plan or "").splitlines():
        bare = line.replace("**", "").strip().lstrip("-• ").strip()
        upper = bare.upper()
        if upper.startswith(_READ_AS_LABEL):
            read_as = read_as or bare[len(_READ_AS_LABEL):].strip()
        elif upper.startswith(_ALSO_ASKED_LABEL):
            text = bare[len(_ALSO_ASKED_LABEL):].strip()
            if text and not _NOTHING_LEFT_OUT.match(text):
                also_asked = also_asked or text
        else:
            body.append(line)
    return read_as, also_asked, "\n".join(body).strip()


def _clean_plan(plan: str) -> str:
    """Remove internal markers and labelled notes from plan text.

    Args:
        plan: The Planner's raw output.

    Returns:
        The numbered plan as a person should read it.
    """
    without_sentinel = _UNANSWERABLE_PREFIX.sub("", plan or "").strip()
    return _split_plan(without_sentinel)[2]


def _restates_question(note: str, question: str) -> bool:
    """Check whether an "also asked" note is just the question again.

    The Planner sometimes repeats a one-part question there. Showing that
    as "not covered" would contradict the answer above it.

    Args:
        note: The Planner's ALSO ASKED text.
        question: The user's question.

    Returns:
        True if the note covers at least 60% of the question's words.
    """
    note_words = set(re.findall(r"[a-z0-9]+", note.lower()))
    question_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    if not note_words or not question_words:
        return False
    return len(note_words & question_words) / len(question_words) >= 0.6


def _plain(text: str) -> str:
    """Escape characters Streamlit would treat as formatting.

    Args:
        text: Text produced by the model.

    Returns:
        The text, safe to place inside a markdown string.
    """
    return str(text).replace("$", "\\$")


def _order_by_column(sql: Optional[str]) -> Optional[str]:
    """Read the column a query sorts by, as a hint for its main measure.

    Args:
        sql: The SQL that produced a result, if known.

    Returns:
        The bare column name of the last ORDER BY, lower-cased, or None.
    """
    matches = _ORDER_BY_PATTERN.findall(sql or "")
    return matches[-1].split(".")[-1].lower() if matches else None


def _screen_columns(df: pd.DataFrame) -> list[Any]:
    """Choose which columns to show on screen.

    Key columns are hidden when a name column already identifies each
    row (for example ``id`` next to ``name``). Results without a name
    column keep their ids, because there the id may be the answer. The
    CSV download always contains every column.

    Args:
        df: The raw query result.

    Returns:
        The columns to display, in their original order.
    """
    lowered = [str(c).lower() for c in df.columns]
    if not any(n == "name" or n.endswith("_name") for n in lowered):
        return list(df.columns)
    kept = [c for c in df.columns if not _is_id_column(c)]
    return kept or list(df.columns)


def _csv_name(question: str) -> str:
    """Build a readable CSV file name from the question.

    Args:
        question: The question that produced the result.

    Returns:
        A file name such as ``total_revenue_by_region.csv``.
    """
    words = re.findall(r"[a-z0-9]+", question.lower())
    stop = {"what", "is", "the", "was", "which", "are", "a", "an", "of", "how", "show", "me"}
    kept = [w for w in words if w not in stop][:6]
    return ("_".join(kept) or "query_result") + ".csv"


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------


def _describe_step(node_name: str, update: dict[str, Any], attempt: int) -> Optional[str]:
    """Turn one graph step into a line for the progress log.

    Args:
        node_name: The LangGraph node that just ran.
        update: The state that node returned.
        attempt: How many times the Programmer has run so far.

    Returns:
        A short markdown line, or None for steps not worth showing.
    """
    if node_name == "planner":
        return "**Planner** checked the question against the database schema."
    if node_name == "handle_unanswerable":
        return (
            "**Planner** found that the data needed isn't in this database, "
            "so it stopped before writing any SQL."
        )
    if node_name == "programmer":
        if attempt <= 1:
            return "**Programmer** wrote the SQL query."
        return f"**Programmer** rewrote the SQL query (attempt {attempt})."
    if node_name == "critic":
        error = update.get("error_message")
        if error:
            return f"**Critic** rejected the query: {_shorten(error)}"
        rows = update.get("row_count")
        if rows is None and isinstance(update.get("execution_result"), pd.DataFrame):
            rows = len(update["execution_result"])
        rows_text = _plural(int(rows), "row") if rows is not None else "the results"
        return f"**Critic** checked and ran the query, which returned {rows_text}."
    if node_name == "visualize":
        chart = _CHART_LABELS.get(str(update.get("chart_type")), "a table")
        return f"**Visualizer** chose {chart}."
    if node_name == "finalize_failure":
        return "Stopped after reaching the repair-attempt limit."
    return None


def run_pipeline(question: str, on_step: Callable[[str], None]) -> dict[str, Any]:
    """Run the question through the graph, reporting each step as it finishes.

    Args:
        question: The user's natural-language question.
        on_step: Called with a markdown line after each meaningful step.

    Returns:
        The final graph state, plus ``attempts`` (SQL attempts) and
        ``nodes`` (the order in which graph nodes ran).
    """
    graph = get_graph()
    initial_state: dict[str, Any] = {
        "question": question,
        "schema_context": get_schema_context(),
        "retry_count": 0,
    }

    final_state: dict[str, Any] = dict(initial_state)
    nodes: list[str] = []
    attempts = 0

    for chunk in graph.stream(initial_state, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        for node_name, update in chunk.items():
            if not isinstance(update, dict):
                continue
            nodes.append(node_name)
            final_state.update(update)
            if node_name == "programmer":
                attempts += 1
            line = _describe_step(node_name, update, attempts)
            if line:
                on_step(line)

    # Safety net: if this LangGraph version streams in an unexpected shape,
    # fall back to a plain invoke rather than showing an empty answer.
    if not nodes:
        logger.warning("App: stream produced no recognisable steps; using invoke().")
        final_state = dict(graph.invoke(initial_state))
        attempts = 1 if final_state.get("sql_query") else 0

    final_state["attempts"] = attempts
    final_state["nodes"] = nodes
    return final_state


def answer_question(question: str) -> dict[str, Any]:
    """Run a question with a live progress panel and package the result.

    Args:
        question: The user's natural-language question.

    Returns:
        A result dict ready to store in chat history and render.
    """
    steps: list[str] = []
    started = time.perf_counter()

    with st.status("Working on your question…", expanded=True) as status:
        # Shown until the first agent reports back, so the panel is never blank.
        status.write("Reading the question and the database schema…")

        def on_step(line: str) -> None:
            steps.append(line)
            status.write(line)

        try:
            result = run_pipeline(question, on_step)
        except Exception as exc:  # noqa: BLE001 - show any failure in the UI
            logger.exception("App: pipeline raised an exception.")
            status.update(label="The pipeline stopped with an error", state="error")
            return {
                "question": question,
                "exception": f"{type(exc).__name__}: {exc}",
                "steps": steps,
                "elapsed": time.perf_counter() - started,
            }
        status.update(label="Done", state="complete", expanded=False)

    result["steps"] = steps
    result["elapsed"] = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# Rendering an answer
# ---------------------------------------------------------------------------


def _trace_text(result: dict[str, Any]) -> str:
    """Summarise the path the question took through the agents.

    Args:
        result: The stored result for this answer.

    Returns:
        Text such as "Planner → Programmer → Critic → Visualizer in 6.4s".
    """
    nodes = result.get("nodes") or []
    labels = [_TRACE_LABELS[n] for n in nodes if n in _TRACE_LABELS]
    if "handle_unanswerable" in nodes:
        labels.append("stopped before writing SQL")
    if "finalize_failure" in nodes:
        labels.append("stopped at the repair limit")
    trace = " → ".join(labels) if labels else "Agent pipeline"
    elapsed = result.get("elapsed")
    if isinstance(elapsed, (int, float)):
        trace += f" in {elapsed:.1f}s"
    return trace


def _display_frame(df: pd.DataFrame) -> Any:
    """Prepare a result for on-screen display with readable numbers.

    The CSV download always uses the raw values; only the screen copy is
    formatted.

    Args:
        df: The raw query result.

    Returns:
        A pandas Styler with thousands separators, or the raw DataFrame if
        styling isn't available.
    """
    floats = [
        c for c in df.columns
        if not _is_id_column(c) and pd.api.types.is_float_dtype(df[c])
    ]
    ints = [
        c for c in df.columns
        if not _is_id_column(c)
        and not _is_time_like(c)
        and pd.api.types.is_integer_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    try:
        styler = df.style
        if floats:
            styler = styler.format("{:,.2f}", subset=floats, na_rep="")
        if ints:
            styler = styler.format("{:,}", subset=ints, na_rep="")
        return styler
    except ImportError:
        return df


def _render_table(df: pd.DataFrame, idx: int) -> None:
    """Show a result table with readable headers and numbers.

    Args:
        df: The raw query result.
        idx: Position of this message in chat history, used for unique keys.
    """
    shown = df[_screen_columns(df)]
    st.dataframe(
        _display_frame(shown),
        hide_index=True,
        column_config={c: _pretty(c) for c in shown.columns},
        key=f"table_{idx}",
    )


def _render_download(df: pd.DataFrame, question: str, idx: int) -> None:
    """Offer the raw result as a CSV download.

    Args:
        df: The raw query result.
        question: The question, used to name the file.
        idx: Position of this message in chat history, used for unique keys.
    """
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=_csv_name(question),
        mime="text/csv",
        icon=":material/download:",
        key=f"csv_{idx}",
    )


def _chart_colors() -> list[str]:
    """Return the chart palette from the theme file, or the built-in default."""
    try:
        configured = st.get_option("theme.chartCategoricalColors")
    except Exception:  # noqa: BLE001 - older Streamlit versions lack this option
        configured = None
    if isinstance(configured, (list, tuple)):
        colors = [str(color) for color in configured if color]
        if colors:
            return colors
    return list(_DEFAULT_CHART_COLORS)


def _render_chart(figure: Any, idx: int) -> None:
    """Show a Plotly figure sized and labelled for a chat answer.

    Args:
        figure: The Visualizer's Plotly figure.
        idx: Position of this message in chat history, used for unique keys.
    """
    traces = list(getattr(figure, "data", []) or [])
    many_series = len(traces) > 1
    colors = _chart_colors()
    for position, trace in enumerate(traces):
        kind = getattr(trace, "type", None)
        if kind not in ("bar", "scatter"):
            continue
        trace.hovertemplate = "%{x}: %{y:,.2f}" + (
            "<extra>%{fullData.name}</extra>" if many_series else "<extra></extra>"
        )
        # Colour each series from the app's palette, whatever the Visualizer
        # or Plotly's defaults chose. A styling problem must never hide a chart.
        color = colors[position % len(colors)]
        try:
            trace.marker.color = color
            if kind == "scatter":
                trace.line.color = color
                # More lines than colours: the second round is dashed.
                if position >= len(colors):
                    trace.line.dash = "dash"
        except Exception:  # noqa: BLE001
            logger.warning("Could not apply the chart colour to a %s series.", kind)
    if hasattr(figure, "update_layout"):
        if many_series:
            figure.update_layout(
                height=400,
                margin={"t": 48, "r": 8},
                showlegend=True,
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
            )
        else:
            figure.update_layout(height=380, margin={"t": 24, "r": 8}, showlegend=False)
    st.plotly_chart(figure, key=f"chart_{idx}", config={"displayModeBar": False})


def _render_metric(df: pd.DataFrame, sql: Optional[str] = None) -> None:
    """Show a one-row result as one or more large numbers.

    Every measure in the row gets its own number, so a result such as
    revenue, expenses and profit is shown in full. The measure the query
    sorts by comes first, because it is the one the question ranks on.

    Args:
        df: A one-row result with at least one numeric column.
        sql: The SQL that produced the result, if known.
    """
    numeric = [
        c for c in df.columns
        if not _is_id_column(c) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric:
        st.dataframe(df, hide_index=True)
        return

    context = [
        str(df[c].iloc[0]) for c in df.columns
        if c not in numeric and not _is_id_column(c) and pd.notna(df[c].iloc[0])
    ]
    if context:
        st.markdown(f"**{_plain(', '.join(context))}**")

    ordered_by = _order_by_column(sql)
    measures = sorted(numeric, key=lambda c: str(c).lower() != ordered_by)[:_MAX_METRICS]
    if len(measures) == 1:
        column = measures[0]
        st.metric(label=_pretty(column), value=_format_number(df[column].iloc[0]))
        return
    for start in range(0, len(measures), 2):
        slots = st.columns(2)
        for slot, column in zip(slots, measures[start:start + 2]):
            with slot:
                st.metric(label=_pretty(column), value=_format_number(df[column].iloc[0]))


def _render_details(result: dict[str, Any], show_plan: bool = True, show_sql: bool = True) -> None:
    """Show the agent steps, plan and SQL behind an answer.

    Args:
        result: The stored result for this answer.
        show_plan: Whether to include the Planner's plan.
        show_sql: Whether to include the SQL.
    """
    with st.expander("How I got this answer", icon=":material/account_tree:"):
        steps = result.get("steps") or []
        if steps:
            st.markdown("**Steps**")
            st.markdown("\n".join(f"- {line}" for line in steps))

        plan = _clean_plan(result.get("query_plan") or "")
        sql = result.get("sql_query")
        if show_plan and plan:
            st.markdown("**Plan**")
            if show_sql and sql:
                st.caption(
                    "The Planner's outline. The Programmer can refine it, so the "
                    "SQL below is the record of what actually ran."
                )
            st.markdown(_plain(plan))

        if show_sql and sql:
            st.markdown("**SQL**")
            st.code(sql, language="sql")

        st.caption(f"Model: {_model_label()}.")


def render_result(result: dict[str, Any], idx: int) -> None:
    """Render one assistant answer.

    Args:
        result: The stored result for this answer.
        idx: Position of this message in chat history, used for unique keys.
    """
    if result.get("exception"):
        st.error(
            "The pipeline stopped with an error, so there's no answer for this "
            "question. The terminal running the app has the full details."
        )
        st.code(result["exception"], language=None)
        return

    st.caption(_trace_text(result))
    final_status = result.get("final_status")
    attempts = int(result.get("attempts") or 0)
    question = str(result.get("question") or "")

    if final_status == "unanswerable":
        reason = _clean_plan(result.get("error_message") or "") or (
            "The data this question needs isn't in the database."
        )
        st.info(
            f"**This question can't be answered from the available data.**\n\n{reason}",
            icon=":material/info:",
        )
        st.caption("No SQL was run, so no number was guessed.")
        second_part = split_question(question)[1]
        if second_part:
            st.warning(
                f"The second part of your question wasn't looked at: \"{_plain(second_part)}\". "
                "Ask it as a separate question.",
                icon=":material/warning:",
            )
        _render_details(result, show_plan=False, show_sql=False)
        return

    if final_status != "success":
        st.error(
            f"I couldn't produce a reliable answer after "
            f"{_plural(attempts, 'attempt')}, so I'm not showing a number.",
            icon=":material/error:",
        )
        if result.get("error_message"):
            st.caption(f"Last problem the Critic found: {_shorten(result['error_message'], 300)}")
        st.caption("Try rephrasing the question, or name the metric and table you mean.")
        _render_details(result)
        return

    df = result.get("execution_result")
    chart_type = result.get("chart_type") or "table"
    figure = result.get("figure")
    has_rows = isinstance(df, pd.DataFrame) and not df.empty
    read_as, also_asked, _ = _split_plan(result.get("query_plan") or "")
    # A note the Planner wrote on its own can be the question repeated; one
    # that comes from a split made in code names a real second part.
    if also_asked and not split_question(question)[1] and _restates_question(also_asked, question):
        also_asked = ""

    if read_as:
        st.caption(f"**Read as:** {_plain(read_as)}")

    if attempts > 1:
        st.caption(
            "The Critic rejected the first query and the Programmer fixed it. "
            "Details are under \"How I got this answer\"."
        )

    if chart_type == "empty" or not has_rows:
        st.info("The query ran correctly but found no matching data.", icon=":material/info:")
    elif chart_type == "metric":
        _render_metric(df, result.get("sql_query"))
    elif figure is not None:
        _render_chart(figure, idx)
    else:
        _render_table(df, idx)
        _render_download(df, question, idx)

    if also_asked:
        if also_asked.rstrip().endswith("?"):
            st.warning(
                "This answer doesn't cover one part of your question. Each question is "
                f"answered on its own, so ask it next as: \"{_plain(also_asked)}\"",
                icon=":material/warning:",
            )
            st.button(
                "Ask this next",
                key=f"also_asked_{idx}",
                icon=":material/arrow_forward:",
                on_click=_queue_question,
                args=(also_asked.strip(),),
            )
        else:
            st.warning(
                f"This answer doesn't cover one part of your question: "
                f"\"{_plain(also_asked)}\". Ask it as a separate question.",
                icon=":material/warning:",
            )

    if has_rows and chart_type != "empty" and (chart_type == "metric" or figure is not None):
        with st.expander(f"View data ({_plural(len(df), 'row')})", icon=":material/table:"):
            _render_table(df, idx)
            _render_download(df, question, idx)

    _render_details(result)


# ---------------------------------------------------------------------------
# Session state callbacks
# ---------------------------------------------------------------------------


def _queue_question(question: str) -> None:
    """Queue an example question to run on the next rerun.

    Args:
        question: The example question that was clicked.
    """
    st.session_state.queued_question = question


def _clear_conversation() -> None:
    """Remove all messages from the conversation."""
    st.session_state.messages = []
    st.session_state.pop("queued_question", None)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def _render_example_buttons(prefix: str) -> None:
    """Show the example questions as full-width buttons.

    Args:
        prefix: Key prefix, so the same buttons can appear in two places.
    """
    for i, (example, icon) in enumerate(EXAMPLE_QUESTIONS):
        st.button(
            example,
            key=f"{prefix}_{i}",
            icon=icon,
            width="stretch",
            on_click=_queue_question,
            args=(example,),
        )
    st.caption(
        "The last one is deliberately unanswerable, to show how the agent "
        "handles questions the data can't support."
    )


def _render_empty_state() -> None:
    """Show how the pipeline works and offer example questions."""
    st.subheader("How it works")
    for row_start in range(0, len(PIPELINE_STEPS), 2):
        columns = st.columns(2)
        for offset, column in enumerate(columns):
            position = row_start + offset
            if position >= len(PIPELINE_STEPS):
                continue
            name, description = PIPELINE_STEPS[position]
            with column:
                with st.container(border=True):
                    st.markdown(f"**{position + 1}. {name}**")
                    st.caption(description)

    st.subheader("Try one of these")
    _render_example_buttons("start_example")


def _parse_schema(schema_text: str) -> list[tuple[str, list[str]]]:
    """Pull table and column names out of the schema description.

    Args:
        schema_text: The SchemaInspector's plain-text schema description.

    Returns:
        (table name, column names) pairs, or an empty list when the text
        isn't in the expected "Table: ... / - column (TYPE)" layout.
    """
    tables: list[tuple[str, list[str]]] = []
    for line in schema_text.splitlines():
        table_match = _SCHEMA_TABLE_LINE.match(line)
        if table_match:
            tables.append((table_match.group(1), []))
            continue
        column_match = _SCHEMA_COLUMN_LINE.match(line)
        if column_match and tables:
            column = column_match.group(1)
            if column not in tables[-1][1]:
                tables[-1][1].append(column)
    if not tables or any(not columns for _, columns in tables):
        return []
    return tables


def _render_schema(schema_text: str) -> None:
    """Show each table with its columns, sized to fit the narrow sidebar.

    Falls back to the raw schema text if it can't be parsed.

    Args:
        schema_text: The SchemaInspector's plain-text schema description.
    """
    tables = _parse_schema(schema_text)
    if not tables:
        st.code(schema_text, language=None)
        return
    for table, columns in tables:
        st.markdown(f"**{table}**")
        st.caption(", ".join(columns))


def _render_sidebar(table_names: list[str], schema_text: str, has_conversation: bool) -> None:
    """Show examples (once a conversation exists), the schema and app notes.

    Args:
        table_names: Names of the tables in the database.
        schema_text: Schema description without sample rows.
        has_conversation: Whether any messages exist yet.
    """
    with st.sidebar:
        if has_conversation:
            st.subheader("Try asking")
            _render_example_buttons("example")

        st.subheader("Database")
        st.markdown(f"`{DATABASE_PATH.name}`, {_plural(len(table_names), 'table')}")
        with st.expander("Tables and columns"):
            _render_schema(schema_text)

        st.subheader("About the answers")
        st.caption(
            f"Model: {_model_label()}, with up to {MAX_SQL_RETRIES} SQL repair "
            "attempts. Every query is read-only and checked automatically before "
            "it runs. For decisions that matter, review the SQL under "
            "\"How I got this answer\"."
        )
        if has_conversation:
            st.button(
                "Clear conversation",
                key="clear_conversation",
                icon=":material/delete:",
                width="stretch",
                on_click=_clear_conversation,
            )


def main() -> None:
    """Lay out the page and handle one round of user input."""
    st.set_page_config(page_title="Ask your data", page_icon="📊", layout="centered")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Ask your data")
    st.caption(
        "Ask about the car wash business in plain English. "
        "Every answer shows the SQL behind it."
    )

    problems = _preflight_problems()
    if problems:
        for problem in problems:
            st.error(problem)
        st.stop()

    queued = st.session_state.pop("queued_question", None)
    typed = st.chat_input("Ask a question about the car wash business")
    question = typed or queued
    has_conversation = bool(st.session_state.messages) or bool(question)

    table_names, schema_text = get_schema_overview()
    _render_sidebar(table_names, schema_text, has_conversation)

    # The start screen lives in one fixed slot. When a question arrives, this
    # slot is emptied straight away, so the start screen can't linger
    # greyed-out under the progress panel while the pipeline runs.
    start_screen = st.empty()
    if not has_conversation:
        with start_screen.container():
            _render_empty_state()
        return

    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                render_result(message["result"], idx)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            result = answer_question(question)
            result["question"] = question
            st.session_state.messages.append({"role": "assistant", "result": result})

        # Rerun so the new answer is drawn by the same code path as history,
        # replacing the live progress panel with the finished answer.
        st.rerun()


main()
