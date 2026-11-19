"""Centralized configuration for the Text-to-SQL Multi-Agent pipeline.

All tunable parameters, paths, and model settings live here so behavior can
be adjusted without touching agent logic.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATABASE_PATH: Path = DATA_DIR / "carwash.db"
DATABASE_URL: str = f"sqlite:///{DATABASE_PATH}"

# ---------------------------------------------------------------------------
# LLM Selection
# ---------------------------------------------------------------------------
# "openai" is the recommended default — Text-to-SQL is highly model-sensitive
# and gpt-4o-mini is cheap, fast, and strong at SQL. Switch to "ollama" only
# for fully-local runs on simple queries.
LLM_PROVIDER: str = "openai"          # Options: "openai" | "ollama"

OPENAI_MODEL: str = "gpt-4o-mini"
OLLAMA_MODEL: str = "llama3"
OLLAMA_BASE_URL: str = "http://localhost:11434"

# Temperature 0.0 → deterministic, reproducible SQL. We do NOT want the model
# being "creative" when generating queries; correctness is everything.
LLM_TEMPERATURE: float = 0.0

# ---------------------------------------------------------------------------
# Agent Behavior
# ---------------------------------------------------------------------------
# Maximum Programmer→Critic retry cycles before failing gracefully.
MAX_SQL_RETRIES: int = 3

# Number of sample rows per table shown to the LLM. Sample data dramatically
# improves Text-to-SQL accuracy by grounding the model in real value formats.
SAMPLE_ROWS_PER_TABLE: int = 3

# Hard cap on rows returned to the UI to avoid rendering huge result sets.
MAX_RESULT_ROWS: int = 1000