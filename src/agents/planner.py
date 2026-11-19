"""The Planner agent — strategy and schema linking.

First stage of the pipeline. Reads the user's question and the full schema
description, then produces a concise natural-language plan naming the exact
tables, columns, joins, filters, and aggregations needed to answer it. The
Planner does NOT write SQL — separating planning from coding measurably
improves final SQL accuracy by letting the model reason about strategy before
syntax.

Besides the numbered plan, the Planner writes two short labelled lines that
the app shows to the user:

* ``READ AS:`` how it interpreted the question (the measure and the period),
  so a vague word such as "profitable" is never silently swapped for
  something else.
* ``ALSO ASKED:`` any part of the question the plan leaves out, so a
  two-part question is never half-answered without saying so.

A question whose second sentence asks for a second result ("... Also
identify which location had the highest total.") is split in code before
the model sees it (see ``src/question_parts.py``). The plan is written for
the first part alone, so the second part can never leak into it as an extra
sort or row limit. The second part is then reworded as a question that
stands on its own and attached as the ``ALSO ASKED`` line.
"""

import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AnalysisState
from src.data_coverage import get_data_coverage, get_latest_data_date
from src.llm import get_llm
from src.logger import get_logger
from src.question_parts import split_question

logger = get_logger(__name__)

_UNANSWERABLE_SENTINEL = "UNANSWERABLE:"
_ALSO_ASKED_LABEL = "ALSO ASKED:"
_MAX_FOLLOW_UP_CHARS = 250
_MIN_FOLLOW_UP_WORDS = 3

_SYSTEM_PROMPT = """\
You are a senior data analyst. Your job is to plan (NOT write) a SQL query.

Given a database schema and a user's question, produce a short, precise plan
that a SQL developer will follow. Use exactly this layout:

  READ AS: <one sentence: the measure you will calculate, how it is defined, and the period it covers>
  1. Which tables are needed.
  2. Which columns are needed from each table.
  3. Any joins required (name the join keys explicitly).
  4. Any filters (WHERE conditions), including date ranges.
  5. Any grouping (GROUP BY) and aggregations (SUM, COUNT, AVG, etc.).
  6. Any sorting or row limits the question implies.
  ALSO ASKED: <only when a part of the question is left out — see the rules>

Rules:
  - Base the plan ONLY on tables and columns that exist in the schema below.
  - If — and only if — the question cannot be answered using the tables and columns in this schema, respond with exactly one line and nothing else:

    UNANSWERABLE: <one clear sentence explaining what data is missing>

    Do not include READ AS or a numbered plan in this case. Do not guess at a related or similar metric as a substitute. Do not suggest a workaround using a different column — if the data isn't there, say so and stop.
  - Measure what was asked. Never swap the requested measure for an easier one. "Profit", "profitable", "profitability", "margin", "earnings" and "surplus" always mean revenue minus expenses, never revenue alone: plan the revenue total, the expense total and their difference, and return all three as columns.
  - If the question uses a word with more than one reasonable meaning ("best", "top", "performance", "busiest"), choose the most direct measure for it and name that measure in READ AS.
  - READ AS is a plain statement of what will be calculated. Example: "READ AS: Profit (total revenue minus total expenses) for each location, all dates." When the question gives no period, the period is "all dates".
  - TODAY for this database is the date given under TODAY below (the latest date in the data), NOT the real current date. Relative periods ("last 12 months", "last quarter", "this year", "recently") are counted back from TODAY. Give the exact start and end dates in step 4 and in READ AS.
  - When the answer combines totals from two tables that each hold many rows per entity (for example transactions and expenses per location, or transactions and employees per location), say in step 3 that each table must be totalled per entity in its own subquery first, and the subquery results then joined. Never plan a direct join between two such tables.
  - When the result is per location, service or employee, return its name as well as its id (join the table that holds the name), so people see "Mumbai Andheri", not "5".
  - A plan describes exactly ONE query that returns ONE table. Never plan a second query, a second result, or a separate lookup. Answer each part of the question with that one table where an extra column or the sort order is enough.
  - Use a row limit only when the question asks for a top or bottom N, or for the single highest or lowest. When it asks for a breakdown ("for each", "per", "by", "broken down by"), return every row.
  - ALSO ASKED is only for a part of the question that the one table cannot show (for example a breakdown by category AND a single "which is highest overall" answer). Plan the first part, then end with one line starting "ALSO ASKED:" followed by the left-out part rewritten as a complete question that makes sense on its own: keep any period or filter from the original and end with a question mark (for example "ALSO ASKED: Which location had the highest total expenses in 2024?"). Never put the whole question there, never list a part the plan does answer, and leave the line out when nothing is left out.
  - Do NOT output SQL. Apart from the READ AS and ALSO ASKED lines, output a numbered plan in plain English.
  - Be concise. No preamble."""

_USER_TEMPLATE = """\
DATABASE SCHEMA:
{schema}

DATA COVERAGE (earliest and latest value of each date column):
{coverage}

TODAY (the latest date in the data):
  {today}

USER QUESTION:
{question}

Produce the query plan."""


_FOLLOW_UP_SYSTEM_PROMPT = """\
You reword one part of a user's question so it can be asked on its own.
Reply with the reworded question only: one line, no quotes, no explanation."""

_FOLLOW_UP_TEMPLATE = """\
FULL QUESTION:
{question}

PART TO REWORD:
{follow_up}

Reword the part as one complete question that makes sense without the rest
of the full question. Keep any period, place or filter from the full
question that applies to it (for example a year). Replace words such as
"it" or "them" with what they refer to. Do not answer it, do not add
anything that was not asked, and end with a question mark."""


def _is_unanswerable(plan: str) -> bool:
    return plan.upper().startswith(_UNANSWERABLE_SENTINEL)


def _usable_follow_up(reply: str) -> Optional[str]:
    """Checks the model's reworded follow-up question.

    Args:
        reply: The raw model reply.

    Returns:
        The question on one line, or None if the reply is not a single
        question of sensible length.
    """
    lines = [line.strip() for line in (reply or "").strip().splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    candidate = lines[0].strip("\"'`* ").strip()
    if candidate.upper().startswith(_ALSO_ASKED_LABEL):
        candidate = candidate[len(_ALSO_ASKED_LABEL):].strip()
    if not candidate.endswith("?") or len(candidate) > _MAX_FOLLOW_UP_CHARS:
        return None
    if len(re.findall(r"[A-Za-z0-9]+", candidate)) < _MIN_FOLLOW_UP_WORDS:
        return None
    return candidate


def _standalone_follow_up(question: str, follow_up: str) -> str:
    """Rewords the left-out part of a question so it stands on its own.

    "identify which location had the highest total expenses" loses the
    year when asked alone, so the model is asked to carry it over. If the
    call fails or the reply is not a single question, the user's own words
    are returned unchanged and without a question mark; the app then shows
    them as text and does not offer to ask them with one click.

    Args:
        question: The full question as the user typed it.
        follow_up: The part the plan leaves out.

    Returns:
        A question ending in "?", or the original wording.
    """
    try:
        response = get_llm().invoke([
            SystemMessage(content=_FOLLOW_UP_SYSTEM_PROMPT),
            HumanMessage(content=_FOLLOW_UP_TEMPLATE.format(question=question, follow_up=follow_up)),
        ])
        reworded = _usable_follow_up(response.content)
    except Exception as exc:  # noqa: BLE001 - the answer must not depend on this call
        logger.warning("Planner: could not reword the follow-up (%s).", exc)
        reworded = None
    return reworded or follow_up


def _attach_follow_up(plan: str, follow_up: str) -> str:
    """Ends the plan with one ALSO ASKED line for the left-out part.

    Args:
        plan: The plan for the main part of the question.
        follow_up: The left-out part, ready to show to the user.

    Returns:
        The plan with any ALSO ASKED line the model wrote itself replaced
        by this one.
    """
    kept = [
        line for line in plan.splitlines()
        if not line.replace("*", "").strip().lstrip("-• ").strip().upper().startswith(_ALSO_ASKED_LABEL)
    ]
    return "\n".join(kept).rstrip() + f"\n{_ALSO_ASKED_LABEL} {follow_up}"


def _tidy_plan(plan: str) -> str:
    """Normalises the model's reply before it enters the pipeline.

    If any line carries the UNANSWERABLE sentinel, the plan is reduced to
    that line alone, so routing never depends on the model having followed
    the "one line and nothing else" instruction exactly.

    Args:
        plan: The raw model reply.

    Returns:
        The cleaned plan text.
    """
    cleaned = plan.strip()
    for line in cleaned.splitlines():
        candidate = line.replace("*", "").replace("`", "").strip().lstrip("-• ").strip()
        if candidate.upper().startswith(_UNANSWERABLE_SENTINEL):
            return candidate
    return cleaned


def plan_query(state: AnalysisState) -> AnalysisState:
    """Generates a query plan from the question and schema.

    Args:
        state: Pipeline state. Requires ``question`` and ``schema_context``.

    Returns:
        The state updated with ``query_plan``.
    """
    logger.info("Planner: analysing question '%.70s'.", state["question"])
    main_part, follow_up = split_question(state["question"])
    if follow_up:
        logger.info("Planner: second part held back as a follow-up: '%.70s'.", follow_up)

    llm = get_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_USER_TEMPLATE.format(
            schema=state["schema_context"],
            coverage=get_data_coverage() or "  (not available)",
            today=get_latest_data_date() or "(not available)",
            question=main_part,
        )),
    ]

    response = llm.invoke(messages)
    plan = _tidy_plan(response.content)
    if follow_up and not _is_unanswerable(plan):
        plan = _attach_follow_up(plan, _standalone_follow_up(state["question"], follow_up))

    logger.info("Planner: plan produced (%d chars).", len(plan))
    return {**state, "query_plan": plan}
