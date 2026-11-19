"""Date coverage of the database: what period the data actually spans.

The demo data stops on a fixed day, so "today" for this database is the
latest date it holds, not the real current date. The Planner and the
Programmer both use these helpers, so that a question such as "the last 12
months" means the latest 12 months of data, on whatever day it is asked.
"""

import re
from functools import lru_cache
from typing import List, Optional, Tuple

from sqlalchemy import create_engine, inspect, text

from src.config import DATABASE_URL
from src.logger import get_logger

logger = get_logger(__name__)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


@lru_cache(maxsize=1)
def _date_ranges() -> Tuple[Tuple[str, str, str, str], ...]:
    """Reads the earliest and latest value of every date column.

    Returns:
        Tuples of (table, column, earliest, latest), cached for the life of
        the process. Empty if the ranges could not be read. Never raises.
    """
    ranges: List[Tuple[str, str, str, str]] = []
    try:
        engine = create_engine(DATABASE_URL)
        try:
            inspector = inspect(engine)
            with engine.connect() as connection:
                for table in inspector.get_table_names():
                    for column in inspector.get_columns(table):
                        name = column["name"]
                        if "date" not in name.lower():
                            continue
                        earliest, latest = connection.execute(
                            text(f'SELECT MIN("{name}"), MAX("{name}") FROM "{table}"')
                        ).one()
                        if earliest is None or latest is None:
                            continue
                        ranges.append((table, name, str(earliest), str(latest)))
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001 - coverage is a hint; never block the pipeline
        logger.warning("Could not read data coverage (%s).", exc)
        return ()
    return tuple(ranges)


def get_data_coverage() -> str:
    """Describes the date range held in each date column of the database.

    Returns:
        One line per date column, such as
        ``  - transactions.date: 2023-09-19 to 2026-09-18``, or an empty
        string if the ranges could not be read.
    """
    return "\n".join(
        f"  - {table}.{column}: {earliest} to {latest}"
        for table, column, earliest, latest in _date_ranges()
    )


def get_latest_data_date() -> Optional[str]:
    """Returns the latest date anywhere in the data, as ``YYYY-MM-DD``.

    This is what the pipeline treats as "today".

    Returns:
        The latest date, or None if no date range could be read.
    """
    latest_dates = [
        latest[:10] for _, _, _, latest in _date_ranges() if _ISO_DATE.match(latest)
    ]
    return max(latest_dates) if latest_dates else None
