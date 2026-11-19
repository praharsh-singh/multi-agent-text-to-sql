"""LangGraph assembly for the Text-to-SQL pipeline.

Wires the Planner, Programmer, and Critic agents into a stateful graph with a
conditional retry loop:

    planner -> programmer -> critic -> (decision)
                   ^                       |
                   |___ retry on error ____|
                                           |
                                           └── success / give-up -> END

The agent functions are imported unchanged from Phase 2. This module only
defines the wiring and the routing logic.
"""

from langgraph.graph import END, START, StateGraph

from src.agents.critic import execute_and_validate
from src.agents.planner import plan_query
from src.agents.visualizer import visualize
from src.agents.programmer import generate_sql
from src.agents.state import AnalysisState
from src.config import MAX_SQL_RETRIES
from src.logger import get_logger

logger = get_logger(__name__)


def _route_after_critic(state: AnalysisState) -> str:
    """Decides where to go after the Critic runs.

    Three outcomes:
      - No error -> the query succeeded -> finish.
      - Error, but retries remain -> go back to the Programmer to repair.
      - Error, retries exhausted -> give up and finish (failed).

    Args:
        state: Current pipeline state after the Critic has run.

    Returns:
        A routing key: "success", "retry", or "give_up".
    """
    if not state.get("error_message"):
        return "success"

    if state.get("retry_count", 0) < MAX_SQL_RETRIES:
        logger.info(
            "Routing: SQL error, retrying (attempt %d of %d).",
            state.get("retry_count", 0) + 1,
            MAX_SQL_RETRIES,
        )
        return "retry"

    logger.warning(
        "Routing: SQL still failing after %d retries — giving up.",
        MAX_SQL_RETRIES,
    )
    return "give_up"


def _finalize_success(state: AnalysisState) -> AnalysisState:
    """Marks the pipeline as successfully completed.

    Args:
        state: Current pipeline state.

    Returns:
        State with ``final_status`` set to "success".
    """
    return {**state, "final_status": "success"}


def _finalize_failure(state: AnalysisState) -> AnalysisState:
    """Marks the pipeline as failed after exhausting retries.

    Args:
        state: Current pipeline state.

    Returns:
        State with ``final_status`` set to "failed".
    """
    logger.error(
        "Pipeline failed. Last error: %s", state.get("error_message", "unknown")
    )
    return {**state, "final_status": "failed"}


_UNANSWERABLE_SENTINEL = "UNANSWERABLE"
_UNANSWERABLE_FALLBACK_SIGNALS = (
    "cannot be answered",
    "can't be answered",
    "cannot be determined from this schema",
    "no table or column",
)


def _is_unanswerable(state: AnalysisState) -> bool:
    """Check whether the Planner has flagged this question as unanswerable.

    Checks the deterministic sentinel first, then falls back to loose
    phrase matching in case the model doesn't follow the exact format.

    Args:
        state: The current graph state, after the planner node has run.

    Returns:
        True if the plan signals the question cannot be answered.
    """
    plan = (state.get("query_plan") or "").strip()
    if _UNANSWERABLE_SENTINEL in plan.upper()[:50]:
        return True
    plan_lower = plan.lower()
    return any(signal in plan_lower for signal in _UNANSWERABLE_FALLBACK_SIGNALS)


def _route_after_planner(state: AnalysisState) -> str:
    """Route straight to graceful handling if the question is unanswerable;
    otherwise proceed to SQL generation.

    Args:
        state: The current graph state, after the planner node has run.

    Returns:
        "unanswerable" or "proceed".
    """
    return "unanswerable" if _is_unanswerable(state) else "proceed"


def handle_unanswerable(state: AnalysisState) -> AnalysisState:
    """Terminal node for questions the Planner has flagged as unanswerable.

    Sets a clean, honest final status without ever invoking the
    Programmer, so no SQL is generated and no number — real or
    fabricated — can reach the user for a question the schema can't
    support.

    Args:
        state: The current graph state, after the planner node has run.

    Returns:
        Updated state with final_status set to "unanswerable" and a
        clear, human-readable reason in error_message.
    """
    plan_text = (state.get("query_plan") or "").strip()
    if plan_text.upper().startswith(_UNANSWERABLE_SENTINEL) and ":" in plan_text:
        reason = plan_text.split(":", 1)[1].strip()
    else:
        reason = plan_text or "This question cannot be answered from the available schema."

    state["final_status"] = "unanswerable"
    state["error_message"] = reason
    state["execution_result"] = None
    state["row_count"] = 0
    return state


def build_graph():
    """Constructs and compiles the Text-to-SQL LangGraph workflow.

    Returns:
        A compiled LangGraph application exposing ``.invoke(state)``.
    """
    workflow = StateGraph(AnalysisState)

    # Register each agent as a node.
    workflow.add_node("planner", plan_query)
    workflow.add_node("programmer", generate_sql)
    workflow.add_node("critic", execute_and_validate)
    workflow.add_node("visualize", visualize)
    workflow.add_node("finalize_success", _finalize_success)
    workflow.add_node("finalize_failure", _finalize_failure)
    workflow.add_node("handle_unanswerable", handle_unanswerable)

    # Start -> planner.
    workflow.add_edge(START, "planner")

    # Conditional branch after the Planner: skip SQL generation entirely
    # if the question can't be answered from this schema.
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "proceed": "programmer",
            "unanswerable": "handle_unanswerable",
        },
    )

    workflow.add_edge("programmer", "critic")

    # Conditional branch after the Critic. A successful query goes
    # through the Visualizer before finalizing.
    workflow.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "success": "visualize",
            "retry": "programmer",
            "give_up": "finalize_failure",
        },
    )

    workflow.add_edge("visualize", "finalize_success")

    # Terminal nodes end the graph.
    workflow.add_edge("finalize_success", END)
    workflow.add_edge("finalize_failure", END)
    workflow.add_edge("handle_unanswerable", END)

    compiled = workflow.compile()
    logger.info("LangGraph workflow compiled.")
    return compiled