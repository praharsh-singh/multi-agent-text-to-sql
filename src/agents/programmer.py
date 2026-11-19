"""The Programmer agent — SQL generation.

Second stage. Takes the Planner's blueprint and writes dialect-correct SQL.
On a retry (when the Critic found an error), it also receives the failed SQL
and the error message, and must produce a corrected query.

Two things are enforced in code after the model replies, so they never
depend on the model following instructions:

* Only the first statement is kept. A two-part question can tempt the model
  into writing two queries, which the Critic would reject outright.
* "Today" is pinned to the latest date in the data. ``date('now')`` would
  count back from the real current date, which drifts further past the end
  of the demo data every day.

When a question has a second part that needs its own result, the Programmer
is shown the first part only (see ``src/question_parts.py``), so the second
part cannot change the query. For the same reason the plan's ``ALSO ASKED``
line, which names the left-out part, is removed before the plan is shown to
the model.
"""

import re
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AnalysisState
from src.data_coverage import get_latest_data_date
from src.llm import get_llm
from src.logger import get_logger
from src.question_parts import split_question

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an expert SQL developer writing queries for a SQLite database.

Write a single, valid SQLite query that fulfils the provided plan and answers
the user's question. Follow these rules strictly:
  - Use ONLY tables and columns that appear in the schema.
  - Write standard SQLite syntax (e.g., use strftime for date handling).
  - CRITICAL - avoiding join fan-out: if the question requires combining an
    aggregate from one child table (e.g. SUM of transactions.total_revenue)
    with an aggregate from a DIFFERENT child table of the same parent (e.g.
    COUNT of employees), do NOT join both child tables directly in the same
    FROM clause. Joining two separate one-to-many relationships together
    creates a cartesian fan-out that silently multiplies SUM() and inflates
    results, even though the query runs without error.
    Instead, compute each child aggregate in its own subquery first, grouped
    by the shared parent key, THEN join the subquery results together. Example
    pattern for "revenue per employee by location":
      SELECT l.id, l.name, rev.total_revenue / emp.employee_count AS metric
      FROM locations l
      JOIN (SELECT location_id, SUM(total_revenue) AS total_revenue
            FROM transactions GROUP BY location_id) rev ON rev.location_id = l.id
      JOIN (SELECT location_id, COUNT(DISTINCT id) AS employee_count
            FROM employees GROUP BY location_id) emp ON emp.location_id = l.id;
  - When counting distinct entities after any join, use COUNT(DISTINCT ...),
    never a plain COUNT().
  - Return every column the plan names. For profit, surplus or margin that
    means the revenue total, the expense total and their difference. When the
    result is per location, service or employee, include its name column.
  - Add LIMIT only when the plan asks for a row limit.
  - Write exactly ONE SQL statement. If the plan or the question asks for two
    different results, write the query for the first one only.
  - Never use 'now', CURRENT_DATE or CURRENT_TIMESTAMP. A TODAY date is given
    with the schema: it is the latest date in the data. Count relative periods
    back from that literal date, for example date('<TODAY>', '-12 months').
  - Return ONLY the SQL query. No explanation, no markdown, no comments.
  - Do not wrap the query in code fences.
  - End the query with a semicolon."""

_INITIAL_TEMPLATE = """\
DATABASE SCHEMA:
{schema}

TODAY (the latest date in the data): {today}

USER QUESTION:
{question}

QUERY PLAN:
{plan}

Write the SQLite query."""

_RETRY_TEMPLATE = """\
DATABASE SCHEMA:
{schema}

TODAY (the latest date in the data): {today}

USER QUESTION:
{question}

QUERY PLAN:
{plan}

Your previous query FAILED. Fix it.

PREVIOUS SQL:
{failed_sql}

ERROR MESSAGE:
{error}

Write a corrected SQLite query."""


def _strip_code_fences(text: str) -> str:
    """Removes markdown code fences if the model added them despite instructions.

    Args:
        text: Raw model output.

    Returns:
        The SQL with surrounding ```sql ... ``` fences removed.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text.strip()


_NOW_LITERAL = re.compile(
    r"""(['"])now\1(?:\s*,\s*(['"])(?:localtime|utc)\2)?""", re.IGNORECASE
)
_CURRENT_KEYWORD = re.compile(r"\bCURRENT_(?:DATE|TIMESTAMP)\b", re.IGNORECASE)
_QUERY_START = re.compile(r"\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_ALSO_ASKED_LABEL = "ALSO ASKED:"
_DROPPED_STATEMENT_NOTE = "ALSO ASKED: a second result that needs its own query"


def _split_statements(sql: str) -> List[str]:
    """Splits SQL text into statements at semicolons outside quotes.

    Args:
        sql: One or more SQL statements.

    Returns:
        The non-empty statements, without their trailing semicolons.
    """
    statements: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    for char in sql:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
            current.append(char)
        elif char == ";":
            statements.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    statements.append("".join(current).strip())
    return [statement for statement in statements if statement]


def _first_statement(sql: str) -> Tuple[str, bool]:
    """Keeps only the first statement of the model's reply.

    Args:
        sql: The SQL the model returned.

    Returns:
        A tuple of (the first statement, ending in a semicolon; whether a
        further query was dropped). Trailing text that is not a query is
        dropped without being reported.
    """
    statements = _split_statements(sql)
    if not statements:
        return sql.strip(), False
    dropped_query = any(_QUERY_START.match(extra) for extra in statements[1:])
    return statements[0] + ";", dropped_query


def _pin_today(sql: str, today: Optional[str]) -> str:
    """Replaces references to the current date with the data's latest date.

    Args:
        sql: A SQL statement.
        today: The latest date in the data as ``YYYY-MM-DD``, or None to
            leave the SQL untouched.

    Returns:
        The SQL with ``'now'``, ``CURRENT_DATE`` and ``CURRENT_TIMESTAMP``
        replaced by the literal date.
    """
    if not today:
        return sql
    literal = f"'{today}'"
    return _CURRENT_KEYWORD.sub(literal, _NOW_LITERAL.sub(literal, sql))


def _plan_without_notes(plan: str) -> str:
    """Removes the ALSO ASKED line from the plan shown to the model.

    That line names a part of the question the query must NOT cover. Left
    in, it can tempt the model into covering it anyway.

    Args:
        plan: The Planner's plan.

    Returns:
        The plan without any ALSO ASKED line.
    """
    kept = [
        line for line in plan.splitlines()
        if not line.replace("*", "").strip().lstrip("-• ").strip().upper().startswith(_ALSO_ASKED_LABEL)
    ]
    return "\n".join(kept).strip()


def generate_sql(state: AnalysisState) -> AnalysisState:
    """Generates or repairs the SQL query.

    On the first pass it writes SQL from the plan. If ``error_message`` is
    present (a retry), it repairs the previous query using the error.

    Args:
        state: Pipeline state. Requires ``question``, ``schema_context``,
               ``query_plan``. Uses ``error_message`` and ``sql_query`` on retry.

    Returns:
        The state updated with ``sql_query`` and an incremented ``retry_count``.
        If a second statement had to be dropped, ``query_plan`` gains an
        ``ALSO ASKED`` line so the app can tell the user.
    """
    is_retry = bool(state.get("error_message"))
    retry_count = state.get("retry_count", 0)
    today = get_latest_data_date()
    question, _ = split_question(state["question"])
    plan_for_model = _plan_without_notes(state["query_plan"])

    if is_retry:
        logger.info("Programmer: repairing SQL (retry %d).", retry_count + 1)
        prompt = _RETRY_TEMPLATE.format(
            schema=state["schema_context"],
            today=today or "(not available)",
            question=question,
            plan=plan_for_model,
            failed_sql=state.get("sql_query", ""),
            error=state["error_message"],
        )
    else:
        logger.info("Programmer: writing initial SQL.")
        prompt = _INITIAL_TEMPLATE.format(
            schema=state["schema_context"],
            today=today or "(not available)",
            question=question,
            plan=plan_for_model,
        )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    sql, dropped = _first_statement(_strip_code_fences(response.content))
    pinned = _pin_today(sql, today)
    if pinned != sql:
        logger.info("Programmer: pinned 'now' to the latest data date (%s).", today)

    plan = state["query_plan"]
    if dropped:
        logger.warning("Programmer: the model wrote more than one statement; kept the first.")
        if _ALSO_ASKED_LABEL not in plan.upper():
            plan = f"{plan}\n{_DROPPED_STATEMENT_NOTE}"

    logger.info("Programmer: SQL produced.")
    return {
        **state,
        "sql_query": pinned,
        "query_plan": plan,
        "retry_count": retry_count + (1 if is_retry else 0),
    }
