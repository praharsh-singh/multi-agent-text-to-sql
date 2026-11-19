"""End-to-end regression check for the Text-to-SQL pipeline.

Run from the project folder:  python test_regression.py

Each check asks the real pipeline a question, then compares its answer
with numbers read straight from the database by a hand-written query.
Nothing is hard-coded, so the checks stay valid if the data changes.

This calls the configured LLM about 50 times (a few cents with
gpt-4o-mini) and takes a couple of minutes. The exit code is 0 only when
every check passes.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.config import DATABASE_PATH
from src.data_coverage import get_data_coverage, get_latest_data_date
from src.graph import build_graph
from src.schema_inspector import SchemaInspector

TOLERANCE = 0.011
SURPLUS_TOLERANCE = 0.021
WHITEFIELD_QUESTION = "What is the average revenue per employee at Whitefield Auto Spa?"


@dataclass
class Outcome:
    """The verdict of one check."""

    passed: bool
    note: str = ""
    problems: List[str] = field(default_factory=list)


class Pipeline:
    """Runs questions through the real graph, as the app does."""

    def __init__(self) -> None:
        self._graph = build_graph()
        self._schema = SchemaInspector().get_schema_description(include_samples=True)

    def ask(self, question: str) -> Dict[str, Any]:
        """Runs one question and returns the final pipeline state.

        Args:
            question: The natural-language question.

        Returns:
            The final state as a plain dict, plus ``rejections``: the
            Critic's reason for each query it sent back.
        """
        state: Dict[str, Any] = {"question": question, "schema_context": self._schema, "retry_count": 0}
        rejections: List[str] = []
        for chunk in self._graph.stream(dict(state), stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                state.update(update)
                if node == "critic" and update.get("error_message"):
                    rejections.append(str(update["error_message"]))
        state["rejections"] = rejections
        return state


def truth(sql: str) -> List[tuple]:
    """Reads the expected answer straight from the database.

    Args:
        sql: A hand-written query.

    Returns:
        The rows it returns.
    """
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _frame(state: Dict[str, Any]) -> Optional[pd.DataFrame]:
    result = state.get("execution_result")
    return result if isinstance(result, pd.DataFrame) else None


def _numbers(df: pd.DataFrame) -> List[float]:
    values: List[float] = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            values.extend(float(v) for v in df[column].dropna())
    return values


def _missing_values(df: pd.DataFrame, expected: List[float], tolerance: float = TOLERANCE) -> List[float]:
    found = _numbers(df)
    return [e for e in expected if not any(abs(e - f) <= tolerance for f in found)]


def _has_text(df: pd.DataFrame, text: str) -> bool:
    wanted = text.lower()
    return any(
        wanted == str(value).lower()
        for column in df.columns
        for value in df[column]
    )


def _note_line(plan: str, label: str) -> str:
    for line in (plan or "").splitlines():
        bare = line.replace("**", "").strip().lstrip("-• ").strip()
        if bare.upper().startswith(label):
            return bare[len(label):].strip()
    return ""


def _restates(note: str, question: str) -> bool:
    """Mirrors the app: a note that repeats the question is not shown."""
    note_words = set(re.findall(r"[a-z0-9]+", note.lower()))
    question_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    if not note_words or not question_words:
        return False
    return len(note_words & question_words) / len(question_words) >= 0.6


def _follow_up_check(state: Dict[str, Any], period: str) -> tuple:
    """Grades the note about a part of the question that was left out.

    Args:
        state: The final pipeline state.
        period: Text the follow-up must keep to mean the same thing when
            asked on its own (for example "2024").

    Returns:
        A tuple of (description for the pass line, list of problems).
    """
    flagged = _note_line(state.get("query_plan") or "", "ALSO ASKED:")
    if not flagged:
        return "", ["the second part of the question is neither answered nor flagged with ALSO ASKED"]
    if not flagged.endswith("?"):
        return f"second part flagged in the user's own words, no one-click follow-up: \"{flagged}\"", []
    if period not in flagged:
        return "", [f"the suggested follow-up lost '{period}', so asking it would cover the wrong period: \"{flagged}\""]
    return f"second part offered as a follow-up: \"{flagged}\"", []


def _expect_success(state: Dict[str, Any]) -> List[str]:
    status = state.get("final_status")
    if status == "unanswerable":
        return [f"refused as unanswerable: {state.get('error_message')}"]
    if status != "success":
        return [f"status was '{status}': {state.get('error_message')}"]
    if _frame(state) is None or _frame(state).empty:
        return ["the query returned no rows"]
    return []


def _values_check(state: Dict[str, Any], expected: List[float], tolerance: float = TOLERANCE) -> Outcome:
    """Every expected number is in the result, and there is one row per number."""
    problems = _expect_success(state)
    if not problems:
        df = _frame(state)
        missing = _missing_values(df, expected, tolerance)
        if missing:
            shown = ", ".join(f"{m:,.2f}" for m in missing[:4])
            problems.append(f"{len(missing)} of {len(expected)} expected numbers missing (e.g. {shown})")
        if len(df) != len(expected):
            problems.append(f"the result has {len(df)} rows; the database gives {len(expected)}")
    count = len(expected)
    note = f"{count} number{'' if count == 1 else 's'} match{'es' if count == 1 else ''} the database, one row each"
    return Outcome(not problems, note, problems)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_region(state: Dict[str, Any]) -> Outcome:
    expected = [r[0] for r in truth(
        "SELECT ROUND(SUM(t.total_revenue), 2) FROM transactions t "
        "JOIN locations l ON l.id = t.location_id GROUP BY l.region"
    )]
    return _values_check(state, expected)


def check_top_services(state: Dict[str, Any]) -> Outcome:
    expected = [r[0] for r in truth(
        "SELECT ROUND(SUM(total_revenue), 2) AS revenue FROM transactions "
        "GROUP BY service_id ORDER BY revenue DESC LIMIT 5"
    )]
    return _values_check(state, expected)


def check_whitefield(state: Dict[str, Any]) -> Outcome:
    expected = [r[0] for r in truth(
        "SELECT (SELECT SUM(t.total_revenue) FROM transactions t WHERE t.location_id = l.id) * 1.0 "
        "/ (SELECT COUNT(*) FROM employees e WHERE e.location_id = l.id) "
        "FROM locations l WHERE l.name = 'Whitefield Auto Spa'"
    )]
    return _values_check(state, expected)


def check_monthly_2024(state: Dict[str, Any]) -> Outcome:
    expected = [r[0] for r in truth(
        "SELECT ROUND(SUM(total_revenue), 2) FROM transactions "
        "WHERE strftime('%Y', date) = '2024' GROUP BY strftime('%Y-%m', date)"
    )]
    return _values_check(state, expected)


def check_satisfaction(state: Dict[str, Any]) -> Outcome:
    problems = []
    if state.get("final_status") != "unanswerable":
        problems.append(f"status was '{state.get('final_status')}', expected a refusal")
    if state.get("sql_query"):
        problems.append("SQL was written for a question the data can't answer")
    return Outcome(not problems, "refused without writing SQL", problems)


def check_three_cities(state: Dict[str, Any]) -> Outcome:
    expected = [r[0] for r in truth(
        "SELECT ROUND(SUM(t.total_revenue), 2) FROM transactions t "
        "JOIN locations l ON l.id = t.location_id "
        "WHERE l.city IN ('Bengaluru', 'Delhi', 'Kolkata') "
        "AND t.date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY l.id"
    )]
    return _values_check(state, expected)


_PROFIT_BY_LOCATION = """
SELECT l.name, ROUND(COALESCE(r.revenue, 0) - COALESCE(e.expenses, 0), 2) AS profit
FROM locations l
LEFT JOIN (SELECT location_id, SUM(total_revenue) AS revenue FROM transactions {where} GROUP BY location_id) r
       ON r.location_id = l.id
LEFT JOIN (SELECT location_id, SUM(amount) AS expenses FROM expenses {where} GROUP BY location_id) e
       ON e.location_id = l.id
ORDER BY profit DESC
"""


def check_most_profitable(state: Dict[str, Any]) -> Outcome:
    top_name, top_profit = truth(_PROFIT_BY_LOCATION.format(where=""))[0]
    problems = _expect_success(state)
    if not problems:
        df = _frame(state)
        if "expenses" not in (state.get("sql_query") or "").lower():
            problems.append("the SQL never uses the expenses table, so this is not profit")
        if _missing_values(df, [top_profit], SURPLUS_TOLERANCE):
            problems.append(f"all-time profit of the top location ({top_profit:,.2f}) is not in the result")
        if not _has_text(df, top_name):
            problems.append(f"the most profitable location ({top_name}) is not in the result")
        flagged = _note_line(state.get("query_plan") or "", "ALSO ASKED:")
        if flagged and not _restates(flagged, state.get("question") or ""):
            problems.append(f"a one-part question was flagged as partly unanswered: \"{flagged}\"")
    return Outcome(not problems, f"{top_name}, profit {top_profit:,.2f}", problems)


def check_surplus_2024(state: Dict[str, Any]) -> Outcome:
    where = "WHERE date BETWEEN '2024-01-01' AND '2024-12-31'"
    expected = [r[1] for r in truth(_PROFIT_BY_LOCATION.format(where=where))]
    return _values_check(state, expected, SURPLUS_TOLERANCE)


def check_expense_breakdown(state: Dict[str, Any]) -> Outcome:
    where = "WHERE date BETWEEN '2024-01-01' AND '2024-12-31'"
    breakdown = [r[0] for r in truth(
        f"SELECT ROUND(SUM(amount), 2) FROM expenses {where} GROUP BY location_id, category"
    )]
    top_name, top_total = truth(
        "SELECT l.name, ROUND(SUM(e.amount), 2) AS total FROM expenses e "
        f"JOIN locations l ON l.id = e.location_id {where.replace('date', 'e.date')} "
        "GROUP BY l.id ORDER BY total DESC LIMIT 1"
    )[0]
    problems = _expect_success(state)
    note = ""
    if not problems:
        df = _frame(state)
        missing = _missing_values(df, breakdown)
        if missing:
            problems.append(f"{len(missing)} of {len(breakdown)} category totals missing")
        if len(df) not in (len(breakdown), len(breakdown) + 1):
            problems.append(f"the result has {len(df)} rows; the database gives {len(breakdown)} location-category totals")
        if not any(_has_text(df, name) for (name,) in truth("SELECT name FROM locations")):
            problems.append("the result shows location ids but no location names")
        if not _missing_values(df, [top_total], SURPLUS_TOLERANCE):
            note = f"breakdown correct; highest total ({top_name}, {top_total:,.2f}) is in the result"
        else:
            follow_up, follow_up_problems = _follow_up_check(state, "2024")
            problems.extend(follow_up_problems)
            note = f"breakdown correct ({len(breakdown)} totals); {follow_up}"
    return Outcome(not problems, note, problems)


def check_two_part_other_wording(state: Dict[str, Any]) -> Outcome:
    outcome = check_region(state)
    problems = list(outcome.problems)
    note = outcome.note
    if not problems:
        follow_up, follow_up_problems = _follow_up_check(state, "2025")
        problems.extend(follow_up_problems)
        note = f"{note}; {follow_up}"
    return Outcome(not problems, note, problems)


def check_region_and_category(state: Dict[str, Any]) -> Outcome:
    expected = [r[0] for r in truth(
        "SELECT ROUND(SUM(t.total_revenue), 2) FROM transactions t "
        "JOIN locations l ON l.id = t.location_id JOIN services s ON s.id = t.service_id "
        "GROUP BY l.region, s.category"
    )]
    return _values_check(state, expected)


def check_last_12_months(state: Dict[str, Any]) -> Outcome:
    names = [r[0] for r in truth(
        "SELECT s.name FROM transactions t JOIN services s ON s.id = t.service_id "
        "WHERE t.date > date((SELECT MAX(date) FROM transactions), '-12 months') "
        "GROUP BY s.id ORDER BY SUM(t.total_revenue) DESC LIMIT 3"
    )]
    problems = _expect_success(state)
    if not problems:
        sql = (state.get("sql_query") or "").lower()
        if "'now'" in sql or "current_date" in sql or "current_timestamp" in sql:
            problems.append("the period is counted back from today, not from the latest date in the data")
        absent = [n for n in names if not _has_text(_frame(state), n)]
        if absent:
            problems.append(f"expected services missing from the result: {', '.join(absent)}")
    return Outcome(not problems, "period anchored on the data; top 3 services match", problems)


def info_bangalore(state: Dict[str, Any]) -> Outcome:
    expected = [r[0] for r in truth(
        "SELECT ROUND(SUM(t.total_revenue), 2) FROM transactions t "
        "JOIN locations l ON l.id = t.location_id WHERE l.city = 'Bengaluru' "
        "AND t.date BETWEEN '2024-01-01' AND '2024-12-31'"
    )]
    per_location = [r[0] for r in truth(
        "SELECT ROUND(SUM(t.total_revenue), 2) FROM transactions t "
        "JOIN locations l ON l.id = t.location_id WHERE l.city = 'Bengaluru' "
        "AND t.date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY l.id"
    )]
    df = _frame(state)
    if state.get("final_status") != "success" or df is None or df.empty or not _numbers(df):
        return Outcome(False, f"returned no data (status '{state.get('final_status')}')")
    if not _missing_values(df, expected) or not _missing_values(df, per_location):
        return Outcome(True, "understood 'Bangalore' as Bengaluru and got the right total")
    return Outcome(False, "returned a number that doesn't match Bengaluru's 2024 revenue")


CHECKS: List[tuple] = [
    ("Total revenue by region",
     "What is the total revenue by region?", check_region),
    ("Top 5 services by revenue",
     "Which 5 services earned the most revenue?", check_top_services),
    ("Revenue per employee at Whitefield",
     WHITEFIELD_QUESTION, check_whitefield),
    ("Monthly revenue in 2024",
     "What was the monthly revenue in 2024?", check_monthly_2024),
    ("Customer satisfaction (must refuse)",
     "What is the average customer satisfaction score?", check_satisfaction),
    ("Three cities, 2024, ranked",
     "For all service centres in Bengaluru, Delhi, and Kolkata, calculate the total revenue "
     "generated by each location between January 1, 2024 and December 31, 2024. Rank the "
     "locations from highest to lowest revenue.", check_three_cities),
    ("Most profitable location (must use expenses)",
     "Which car-wash location is the most profitable?", check_most_profitable),
    ("Operating surplus by location, 2024",
     "Use 2024 data. Define operating surplus as total transaction revenue minus total recorded "
     "operating expenses. Calculate this separately for every location, rank the locations, and "
     "show revenue, expenses, and the resulting operating surplus.", check_surplus_2024),
    ("Expenses by category, plus highest location",
     "For each service centre, show the total operating expenses for 2024, broken down by "
     "expense category. Also identify which location had the highest total expenses.",
     check_expense_breakdown),
    ("Two-part question, different wording",
     "What is the total revenue by region? Also, which service brought in the most revenue "
     "in 2025?", check_two_part_other_wording),
    ("Revenue by region and service category (three tables)",
     "What is the total revenue by region and service category?", check_region_and_category),
    ("Top 3 services, last 12 months",
     "Which 3 services generated the most revenue in the last 12 months?", check_last_12_months),
]

INFO_CHECKS: List[tuple] = [
    ("City spelled 'Bangalore'",
     "What was the total revenue of the Bangalore locations in 2024?", info_bangalore),
]


def _print_failure(state: Dict[str, Any], outcome: Outcome) -> None:
    for problem in outcome.problems:
        print(f"         why    : {problem}")
    read_as = _note_line(state.get("query_plan") or "", "READ AS:")
    if read_as:
        print(f"         read as: {read_as}")
    also_asked = _note_line(state.get("query_plan") or "", "ALSO ASKED:")
    if also_asked:
        print(f"         also asked: {also_asked}")
    sql = " ".join((state.get("sql_query") or "(none)").split())
    print(f"         sql    : {sql[:600]}")
    df = _frame(state)
    if df is not None and not df.empty:
        for line in df.head(5).to_string(index=False).splitlines():
            print(f"         result : {line}")


def _print_rejections(state: Dict[str, Any]) -> None:
    """Shows why the Critic sent a query back, even when the answer was right."""
    for number, reason in enumerate(state.get("rejections") or [], start=1):
        print(f"         attempt {number} sent back: {' '.join(reason.split())[:260]}")


def _run(pipeline: Pipeline, question: str, check: Callable[[Dict[str, Any]], Outcome]):
    try:
        state = pipeline.ask(question)
        return state, check(state)
    except Exception as exc:  # noqa: BLE001 - one broken check must not stop the rest
        return {}, Outcome(False, "", [f"{type(exc).__name__}: {exc}"])


def main() -> int:
    """Runs every check and prints a pass/fail summary.

    Returns:
        0 if all graded checks pass, otherwise 1.
    """
    logging.disable(logging.WARNING)

    coverage = get_data_coverage()
    print("Data coverage given to the Planner:")
    print(coverage or "  (none - the date ranges could not be read)")
    print(f"Treated as today: {get_latest_data_date() or '(unknown)'}")
    print()

    pipeline = Pipeline()
    passed = 0
    total = 0
    with_read_as = 0
    answered = 0
    rejections = 0

    for name, question, check in CHECKS:
        state, outcome = _run(pipeline, question, check)
        total += 1
        passed += outcome.passed
        retries = state.get("retry_count", 0) or 0
        rejections += retries
        if state.get("final_status") == "success":
            answered += 1
            with_read_as += bool(_note_line(state.get("query_plan") or "", "READ AS:"))
        tail = f" ({retries} retr{'y' if retries == 1 else 'ies'})" if state else ""
        print(f"[{'PASS' if outcome.passed else 'FAIL'}] {name}{tail}")
        if outcome.passed and outcome.note:
            print(f"         {outcome.note}")
        if not outcome.passed:
            _print_failure(state, outcome)
        _print_rejections(state)

    print("\nStability: the Whitefield question three more times")
    stable = True
    for attempt in range(1, 4):
        state, outcome = _run(pipeline, WHITEFIELD_QUESTION, check_whitefield)
        stable = stable and outcome.passed
        retries = state.get("retry_count", 0) or 0
        print(f"  run {attempt}: {'correct' if outcome.passed else 'WRONG OR FAILED'}, {retries} retries")
        if not outcome.passed:
            _print_failure(state, outcome)
        _print_rejections(state)
    total += 1
    passed += stable
    print(f"[{'PASS' if stable else 'FAIL'}] Whitefield answer is stable across runs")

    print("\nFor information (not graded):")
    for name, question, check in INFO_CHECKS:
        state, outcome = _run(pipeline, question, check)
        print(f"  {name}: {outcome.note or '; '.join(outcome.problems)}")
        read_as = _note_line(state.get("query_plan") or "", "READ AS:")
        if read_as:
            print(f"         read as: {read_as}")
    print(f"  READ AS line present on {with_read_as} of {answered} answered questions")
    print(f"  Critic rejections across the graded questions: {rejections}")

    print(f"\n{passed}/{total} passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
