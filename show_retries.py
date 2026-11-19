"""Diagnostic: show every SQL attempt and the Critic's verdict on it.

Run from the project folder:  python show_retries.py

For each question this prints each query the Programmer wrote and either
the Critic's reason for rejecting it or the number of rows it returned.
It exists to answer one question: when the Critic sends a query back, was
that query actually wrong?
"""
from __future__ import annotations

import logging
import sys

from src.graph import build_graph
from src.schema_inspector import SchemaInspector

QUESTIONS = [
    "What is the average revenue per employee at Whitefield Auto Spa?",
    "What is the total revenue by region and service category?",
]


def main() -> int:
    """Runs each question once and prints every attempt.

    Returns:
        Always 0; this script reports, it does not grade.
    """
    logging.disable(logging.WARNING)
    graph = build_graph()
    schema = SchemaInspector().get_schema_description(include_samples=True)

    for question in QUESTIONS:
        print("=" * 78)
        print(f"QUESTION: {question}")
        attempt = 0
        state = {"question": question, "schema_context": schema, "retry_count": 0}
        for chunk in graph.stream(state, stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                if node == "planner":
                    print("\n--- PLAN ---")
                    print(update.get("query_plan"))
                elif node == "programmer":
                    attempt += 1
                    print(f"\n--- ATTEMPT {attempt}: SQL ---")
                    print(update.get("sql_query"))
                elif node == "critic":
                    error = update.get("error_message")
                    if error:
                        print(f"--- ATTEMPT {attempt}: REJECTED ---")
                        print(error)
                    else:
                        print(f"--- ATTEMPT {attempt}: ACCEPTED, {update.get('row_count')} row(s) ---")
                        result = update.get("execution_result")
                        if result is not None:
                            print(result.head(8).to_string(index=False))
                elif node in ("finalize_failure", "handle_unanswerable"):
                    print(f"\n--- ENDED AT: {node} ---")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
