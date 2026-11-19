"""Reads the SQLite schema and renders it as an LLM-friendly description.

The quality of Text-to-SQL depends heavily on how the schema is presented to
the model.  This module produces a compact, structured description including
column types, primary/foreign keys, and a few sample rows per table — a
combination proven to significantly improve query-generation accuracy.
"""

from typing import List

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from src.config import DATABASE_URL, SAMPLE_ROWS_PER_TABLE
from src.logger import get_logger

logger = get_logger(__name__)


class SchemaInspector:
    """Introspects the database and builds a text schema for the agents.

    Attributes:
        engine: The SQLAlchemy engine bound to the configured database.

    Example:
        >>> inspector = SchemaInspector()
        >>> print(inspector.get_schema_description())
    """

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        """Initializes the inspector and opens a database connection.

        Args:
            database_url: SQLAlchemy connection URL. Defaults to config value.
        """
        self.engine: Engine = create_engine(database_url)

    def get_table_names(self) -> List[str]:
        """Returns all user table names in the database.

        Returns:
            A list of table name strings.
        """
        return inspect(self.engine).get_table_names()

    def _format_columns(self, table_name: str) -> str:
        """Formats one table's columns, marking primary and foreign keys.

        Args:
            table_name: The table to describe.

        Returns:
            A newline-indented column description block.
        """
        insp = inspect(self.engine)
        pk_cols = set(insp.get_pk_constraint(table_name).get("constrained_columns", []))

        # Map column name -> "referenced_table.referenced_column".
        fk_map = {}
        for fk in insp.get_foreign_keys(table_name):
            for local, remote in zip(fk["constrained_columns"],
                                     fk["referred_columns"]):
                fk_map[local] = f"{fk['referred_table']}.{remote}"

        lines = []
        for col in insp.get_columns(table_name):
            tags = []
            if col["name"] in pk_cols:
                tags.append("PRIMARY KEY")
            if col["name"] in fk_map:
                tags.append(f"FOREIGN KEY -> {fk_map[col['name']]}")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"    - {col['name']} ({col['type']}){tag_str}")
        return "\n".join(lines)

    def _get_sample_rows(self, table_name: str) -> str:
        """Fetches a few sample rows to ground the model in real value formats.

        Args:
            table_name: The table to sample.

        Returns:
            A formatted sample-rows block, or a note if the table is empty.
        """
        query = text(f"SELECT * FROM {table_name} LIMIT :n")
        with self.engine.connect() as conn:
            result = conn.execute(query, {"n": SAMPLE_ROWS_PER_TABLE})
            rows = result.fetchall()

        if not rows:
            return "    (no rows)"
        return "\n".join(f"    {tuple(row)}" for row in rows)

    def get_schema_description(self, include_samples: bool = True) -> str:
        """Builds the complete schema description string for the LLM.

        Args:
            include_samples: Whether to append sample rows per table.

        Returns:
            A single formatted string describing every table, its columns,
            key relationships, and (optionally) sample data.
        """
        blocks: List[str] = []
        for table_name in self.get_table_names():
            block = [f"Table: {table_name}", "  Columns:"]
            block.append(self._format_columns(table_name))
            if include_samples:
                block.append("  Sample rows:")
                block.append(self._get_sample_rows(table_name))
            blocks.append("\n".join(block))

        description = "\n\n".join(blocks)
        logger.info(
            "Schema description built for %d table(s).",
            len(self.get_table_names()),
        )
        return description


if __name__ == "__main__":
    inspector = SchemaInspector()
    print("\n" + "=" * 60)
    print("  DATABASE SCHEMA")
    print("=" * 60 + "\n")
    print(inspector.get_schema_description())