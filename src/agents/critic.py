"""The Critic agent — validation and execution.

Third stage. Before executing, it runs cheap static checks that catch the
semantic fan-out bugs plain execution cannot (a valid-but-wrong query runs
fine and returns a plausible number). If a check fires, the query is treated
like an execution error so the existing retry loop routes it back to the
Programmer for repair. If the checks pass, the SQL is executed against the
database, and the result gets one more check before it is accepted.

Only read-only SELECT queries are permitted, and SQLite databases are opened
read-only, so a malformed or malicious generation can never change the data.

How the fan-out check works
---------------------------
A join only inflates a total when it goes "down" a one-to-many relationship.
Joining transactions to their location (many transactions, one location) is
safe: each transaction still appears once. Joining a location to its
transactions AND its employees is not: each transaction is repeated once per
employee, so SUM(total_revenue) is multiplied.

So instead of counting joins, the check reads each ON condition. A condition
such as ``t.location_id = l.id`` matches a primary key on ``l``, so each row
of ``t`` meets at most one row of ``l``. A query is safe when some table can
reach every other table through such many-to-one steps (that table sets the
grain of the result) and every SUM/AVG/COUNT adds up columns of that table.
Subqueries and CTEs are checked on their own, with GROUP BY telling which of
their columns are unique.

The check is conservative where it can't tell: if the SQL is in a shape it
doesn't parse, it lets the query run rather than rejecting a correct one.
"""

import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import NullPool

from src.agents.state import AnalysisState
from src.config import DATABASE_PATH, DATABASE_URL, MAX_RESULT_ROWS
from src.logger import get_logger

logger = get_logger(__name__)

_FORBIDDEN = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
})

# Aggregates whose result changes when rows are repeated. MIN and MAX don't.
_SUMMING_AGGREGATES = frozenset({"SUM", "AVG", "COUNT", "TOTAL", "GROUP_CONCAT"})

_CLAUSE_WORDS = frozenset({"FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "WINDOW"})
_JOIN_WORDS = frozenset({"JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL"})
_SET_OPERATORS = frozenset({"UNION", "INTERSECT", "EXCEPT"})
_NOT_ALIASES = _CLAUSE_WORDS | _JOIN_WORDS | _SET_OPERATORS | frozenset({"ON", "USING", "AS"})
_NOT_COLUMNS = frozenset({
    "SELECT", "DISTINCT", "ALL", "AS", "CASE", "WHEN", "THEN", "ELSE", "END",
    "AND", "OR", "NOT", "NULL", "IS", "IN", "LIKE", "GLOB", "BETWEEN", "EXISTS",
    "CAST", "INTEGER", "INT", "REAL", "TEXT", "NUMERIC", "FLOAT", "TRUE", "FALSE",
    "FILTER", "OVER", "PARTITION", "BY", "ORDER", "ASC", "DESC", "ESCAPE",
    "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP", "COLLATE", "NOCASE",
})


# ---------------------------------------------------------------------------
# Tokenizing and nesting
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(
    r"""
      (?P<space>\s+)
    | (?P<comment>--[^\n]*|/\*.*?\*/)
    | (?P<string>'(?:[^']|'')*')
    | (?P<quoted>"(?:[^"]|"")*"|`[^`]*`|\[[^\]]*\])
    | (?P<number>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+)
    | (?P<word>[A-Za-z_][A-Za-z0-9_$]*)
    | (?P<symbol><=|>=|<>|!=|==|\|\||[(),.;=<>+\-*/%])
    | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True)
class _Token:
    """One lexical token. ``kind`` is word, quoted, string, number, symbol or other."""

    kind: str
    text: str

    @property
    def upper(self) -> str:
        return self.text.upper() if self.kind == "word" else ""

    @property
    def name(self) -> Optional[str]:
        """The identifier this token names, lower-cased, or None."""
        if self.kind == "word":
            return self.text.lower()
        if self.kind == "quoted":
            return self.text[1:-1].lower()
        return None


class _Group(list):
    """The tokens inside one pair of parentheses."""


_Item = Union[_Token, _Group]


def _tokenize(sql: str) -> List[_Token]:
    tokens = []
    for match in _TOKEN_PATTERN.finditer(sql):
        kind = match.lastgroup
        if kind in ("space", "comment"):
            continue
        tokens.append(_Token(kind, match.group()))
    return tokens


def _nest(tokens: Sequence[_Token]) -> Optional[List[_Item]]:
    """Folds parenthesised runs into _Group items. None if unbalanced."""
    stack: List[List[_Item]] = [[]]
    for token in tokens:
        if token.text == "(":
            group = _Group()
            stack[-1].append(group)
            stack.append(group)
        elif token.text == ")":
            if len(stack) == 1:
                return None
            stack.pop()
        else:
            stack[-1].append(token)
    return stack[0] if len(stack) == 1 else None


def _word(item: _Item) -> str:
    return item.upper if isinstance(item, _Token) else ""


def _is_query_group(item: _Item) -> bool:
    return isinstance(item, _Group) and bool(item) and _word(item[0]) in ("SELECT", "WITH")


def _split(items: Sequence[_Item], separator: str) -> List[List[_Item]]:
    parts: List[List[_Item]] = [[]]
    for item in items:
        if isinstance(item, _Token) and item.text == separator:
            parts.append([])
        else:
            parts[-1].append(item)
    return parts


def _split_on_word(items: Sequence[_Item], words: FrozenSet[str]) -> List[List[_Item]]:
    parts: List[List[_Item]] = [[]]
    for item in items:
        if _word(item) in words:
            parts.append([])
        else:
            parts[-1].append(item)
    return parts


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

# table name -> (column names, primary key columns), all lower-case
_Schema = Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]


@lru_cache(maxsize=1)
def _database_schema() -> _Schema:
    """Reads every table's columns and primary key. Empty on any failure."""
    schema: _Schema = {}
    try:
        engine = create_engine(DATABASE_URL)
        try:
            inspector = inspect(engine)
            for table in inspector.get_table_names():
                columns = frozenset(c["name"].lower() for c in inspector.get_columns(table))
                primary = inspector.get_pk_constraint(table) or {}
                keys = frozenset(c.lower() for c in primary.get("constrained_columns") or [])
                schema[table.lower()] = (columns, keys)
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001 - the check degrades, it never blocks
        logger.warning("Critic: could not read the schema for fan-out checks (%s).", exc)
        return {}
    return schema


# ---------------------------------------------------------------------------
# Scope analysis
# ---------------------------------------------------------------------------


@dataclass
class _Source:
    """One table, subquery or CTE in a FROM clause."""

    label: str
    columns: Set[str] = field(default_factory=set)
    unique_sets: List[FrozenSet[str]] = field(default_factory=list)


@dataclass
class _Edge:
    """The join conditions between two sources, and which side is unique."""

    columns: Dict[str, Set[str]] = field(default_factory=dict)
    unique: Dict[str, bool] = field(default_factory=dict)


class _Unparsed(Exception):
    """The SQL is in a shape the analysis doesn't understand."""


class _Scope:
    """Analyses one SELECT and the queries nested inside it."""

    def __init__(self, schema: _Schema, ctes: Dict[str, _Source], problems: List[str]):
        self.schema = schema
        self.ctes = dict(ctes)
        self.problems = problems
        self.sources: Dict[str, _Source] = {}
        self.order: List[str] = []
        self.using: List[Tuple[str, List[str]]] = []

    # -- statements ---------------------------------------------------------

    def statement(self, items: List[_Item]) -> _Source:
        """Analyses a full query, with any WITH clause. Returns its output shape."""
        if _word(items[0]) == "WITH":
            items = self._read_ctes(items[1:])
        parts = _split_on_word(items, _SET_OPERATORS)
        outputs = []
        for part in parts:
            if part and _word(part[0]) == "ALL":
                part = part[1:]
            if not part or _word(part[0]) != "SELECT":
                raise _Unparsed("compound part is not a SELECT")
            outputs.append(_Scope(self.schema, self.ctes, self.problems).select(part))
        if len(outputs) == 1:
            return outputs[0]
        return _Source(label="compound query", columns=set(outputs[0].columns))

    def _read_ctes(self, items: List[_Item]) -> List[_Item]:
        if _word(items[0]) == "RECURSIVE":
            raise _Unparsed("recursive CTE")
        position = 0
        while True:
            name_token = items[position]
            if not isinstance(name_token, _Token) or name_token.name is None:
                raise _Unparsed("CTE name")
            position += 1
            if isinstance(items[position], _Group):
                raise _Unparsed("CTE column list")
            if _word(items[position]) != "AS":
                raise _Unparsed("CTE without AS")
            position += 1
            if _word(items[position]) in ("MATERIALIZED", "NOT"):
                raise _Unparsed("CTE hint")
            body = items[position]
            if not _is_query_group(body):
                raise _Unparsed("CTE body")
            output = _Scope(self.schema, self.ctes, self.problems).statement(list(body))
            output.label = name_token.name
            self.ctes[name_token.name] = output
            position += 1
            if isinstance(items[position], _Token) and items[position].text == ",":
                position += 1
                continue
            return items[position:]

    # -- one SELECT ---------------------------------------------------------

    def select(self, items: List[_Item]) -> _Source:
        clauses = self._clauses(items)
        select_items = clauses["SELECT"]
        distinct = bool(select_items) and _word(select_items[0]) == "DISTINCT"
        if select_items and _word(select_items[0]) in ("DISTINCT", "ALL"):
            select_items = select_items[1:]

        if clauses.get("FROM"):
            self._read_from(clauses["FROM"])
        self._visit_nested(items)

        edges = self._edges(clauses.get("FROM", []), clauses.get("WHERE", []))
        roots = self._roots(edges)
        for part in ("SELECT", "HAVING", "ORDER"):
            self._check_aggregates(clauses.get(part, []), edges, roots)

        return self._output(select_items, clauses.get("GROUP", []), distinct)

    def _clauses(self, items: List[_Item]) -> Dict[str, List[_Item]]:
        clauses: Dict[str, List[_Item]] = {"SELECT": []}
        current = "SELECT"
        position = 1
        while position < len(items):
            word = _word(items[position])
            if word in _CLAUSE_WORDS:
                current = word
                clauses.setdefault(current, [])
                position += 1
                if word in ("GROUP", "ORDER") and _word(items[position] if position < len(items) else None) == "BY":
                    position += 1
                continue
            clauses[current].append(items[position])
            position += 1
        return clauses

    def _visit_nested(self, items: Sequence[_Item]) -> None:
        """Analyses subqueries that aren't FROM sources (scalar, IN, EXISTS)."""
        for item in items:
            if isinstance(item, _Group):
                if _is_query_group(item) and not getattr(item, "_seen", False):
                    _Scope(self.schema, self.ctes, self.problems).statement(list(item))
                elif not _is_query_group(item):
                    self._visit_nested(item)

    # -- FROM ---------------------------------------------------------------

    def _read_from(self, items: List[_Item]) -> None:
        position = 0
        while position < len(items):
            item = items[position]
            if _word(item) in _JOIN_WORDS:
                if _word(item) == "NATURAL":
                    raise _Unparsed("natural join")
                position += 1
                continue
            if isinstance(item, _Token) and item.text == ",":
                position += 1
                continue
            if _word(item) == "ON":
                position = self._skip_condition(items, position + 1)
                continue
            if _word(item) == "USING":
                group = items[position + 1] if position + 1 < len(items) else None
                if not isinstance(group, _Group):
                    raise _Unparsed("USING without columns")
                self.using.append((self.order[-1], [t.name for t in group if isinstance(t, _Token) and t.name]))
                position += 2
                continue
            position = self._read_source(items, position)

    def _skip_condition(self, items: List[_Item], position: int) -> int:
        while position < len(items):
            item = items[position]
            if _word(item) in _JOIN_WORDS or (isinstance(item, _Token) and item.text == ","):
                return position
            position += 1
        return position

    def _read_source(self, items: List[_Item], position: int) -> int:
        item = items[position]
        if isinstance(item, _Group):
            if not _is_query_group(item):
                raise _Unparsed("parenthesised join")
            item._seen = True  # type: ignore[attr-defined]
            source = _Scope(self.schema, self.ctes, self.problems).statement(list(item))
            default_alias = None
            position += 1
        elif isinstance(item, _Token) and item.name:
            parts = [item.name]
            position += 1
            while (position + 1 < len(items) and isinstance(items[position], _Token)
                   and items[position].text == "." and isinstance(items[position + 1], _Token)):
                parts.append(items[position + 1].name or "")
                position += 2
            table = parts[-1]
            source = self._table_source(table)
            default_alias = table
        else:
            raise _Unparsed("FROM item")

        alias = default_alias
        if position < len(items) and _word(items[position]) == "AS":
            position += 1
        if (position < len(items) and isinstance(items[position], _Token)
                and items[position].name and _word(items[position]) not in _NOT_ALIASES):
            alias = items[position].name
            position += 1
        if alias is None:
            alias = f"subquery {len(self.order) + 1}"
        if default_alias is None:
            source.label = f"subquery {alias}"
        if alias in self.sources:
            raise _Unparsed("repeated alias")
        self.sources[alias] = source
        self.order.append(alias)
        return position

    def _table_source(self, table: str) -> _Source:
        if table in self.ctes:
            cte = self.ctes[table]
            return _Source(cte.label, set(cte.columns), list(cte.unique_sets))
        columns, keys = self.schema.get(table, (frozenset(), frozenset()))
        unique = [frozenset(keys)] if keys else [frozenset({"id"})]
        return _Source(table, set(columns), unique)

    # -- columns ------------------------------------------------------------

    def _refs(self, items: Sequence[_Item]) -> Iterator[Tuple[Optional[str], str]]:
        """Yields (qualifier or None, column) for column references in items."""
        position = 0
        while position < len(items):
            item = items[position]
            if isinstance(item, _Group):
                if not _is_query_group(item):
                    yield from self._refs(item)
                position += 1
                continue
            name = item.name
            following = items[position + 1] if position + 1 < len(items) else None
            preceding = items[position - 1] if position > 0 else None
            if name is None or (isinstance(preceding, _Token) and preceding.text == "."):
                position += 1
                continue
            if isinstance(following, _Token) and following.text == ".":
                after = items[position + 2] if position + 2 < len(items) else None
                if isinstance(after, _Token) and after.name:
                    yield name, after.name
                    position += 3
                    continue
                position += 1
                continue
            if isinstance(following, _Group) or _word(item) in _NOT_COLUMNS:
                position += 1
                continue
            yield None, name
            position += 1

    def _resolve(self, qualifier: Optional[str], column: str) -> Optional[str]:
        if qualifier is not None:
            return qualifier if qualifier in self.sources else None
        owners = [alias for alias, source in self.sources.items() if column in source.columns]
        return owners[0] if len(owners) == 1 else None

    # -- joins --------------------------------------------------------------

    def _edges(self, from_items: List[_Item], where_items: List[_Item]) -> Dict[FrozenSet[str], _Edge]:
        edges: Dict[FrozenSet[str], _Edge] = {}

        def add(left: str, left_column: str, right: str, right_column: str) -> None:
            if left == right:
                return
            edge = edges.setdefault(frozenset({left, right}), _Edge())
            edge.columns.setdefault(left, set()).add(left_column)
            edge.columns.setdefault(right, set()).add(right_column)

        conjuncts = list(self._conjuncts(from_items, capturing=False))
        conjuncts += list(self._conjuncts(where_items, capturing=True))
        for conjunct in conjuncts:
            if len(conjunct) < 3:
                continue
            refs = list(self._refs(conjunct))
            equals = [i for i, t in enumerate(conjunct) if isinstance(t, _Token) and t.text in ("=", "==")]
            if len(refs) != 2 or len(equals) != 1:
                continue
            ends = [self._resolve(q, c) for q, c in refs]
            if None in ends:
                continue
            add(ends[0], refs[0][1], ends[1], refs[1][1])

        for alias, columns in self.using:
            for column in columns:
                for other in self.order:
                    if other != alias and column in self.sources[other].columns:
                        add(alias, column, other, column)
                        break

        for pair, edge in edges.items():
            for alias in pair:
                used = edge.columns.get(alias, set())
                edge.unique[alias] = any(u and u <= used for u in self.sources[alias].unique_sets)

        self._connect_components(edges)
        return edges

    @staticmethod
    def _conjuncts(items: List[_Item], capturing: bool) -> Iterator[List[_Item]]:
        """Yields AND-separated conditions: from ON clauses, or a whole WHERE.

        A condition joined by OR is skipped, since it doesn't force a match.
        """
        current: List[_Item] = []
        has_or = False
        between = 0
        for item in items:
            word = _word(item)
            if word == "ON":
                if capturing and current and not has_or:
                    yield current
                current, has_or, capturing = [], False, True
                continue
            if word in _JOIN_WORDS or (isinstance(item, _Token) and item.text == ","):
                if capturing and current and not has_or:
                    yield current
                current, has_or, capturing = [], False, False
                continue
            if not capturing:
                continue
            if word == "BETWEEN":
                between += 1
            if word == "AND":
                if between:
                    between -= 1
                    current.append(item)
                    continue
                if current and not has_or:
                    yield current
                current, has_or = [], False
                continue
            if word == "OR":
                has_or = True
            current.append(item)
        if capturing and current and not has_or:
            yield current

    def _connect_components(self, edges: Dict[FrozenSet[str], _Edge]) -> None:
        """Sources with no join condition between them form a cross product."""
        remaining = list(self.order)
        components: List[Set[str]] = []
        while remaining:
            start = remaining.pop(0)
            component, frontier = {start}, [start]
            while frontier:
                node = frontier.pop()
                for pair in edges:
                    if node in pair:
                        (other,) = pair - {node}
                        if other not in component:
                            component.add(other)
                            frontier.append(other)
            remaining = [a for a in remaining if a not in component]
            components.append(component)
        for first, second in zip(components, components[1:]):
            a, b = sorted(first)[0], sorted(second)[0]
            edges[frozenset({a, b})] = _Edge(unique={a: False, b: False})

    def _roots(self, edges: Dict[FrozenSet[str], _Edge]) -> Set[str]:
        """Sources whose rows each appear once in the joined result."""
        roots = set()
        for start in self.order:
            visited, frontier, ok = {start}, [start], True
            while frontier and ok:
                node = frontier.pop()
                for pair, edge in edges.items():
                    if node not in pair:
                        continue
                    (other,) = pair - {node}
                    if other in visited:
                        continue
                    if not edge.unique.get(other, False):
                        ok = False
                        break
                    visited.add(other)
                    frontier.append(other)
            if ok:
                roots.add(start)
        return roots

    def _multiplier(self, alias: Optional[str], edges: Dict[FrozenSet[str], _Edge]) -> str:
        """Names a source whose join repeats rows of ``alias`` (or of the result)."""
        if alias is not None:
            for pair, edge in edges.items():
                if alias in pair:
                    (other,) = pair - {alias}
                    if not edge.unique.get(other, False):
                        return self.sources[other].label
        for pair, edge in edges.items():
            for node in pair:
                (other,) = pair - {node}
                if other != alias and not edge.unique.get(other, False):
                    return self.sources[other].label
        return "another table"

    # -- aggregates ---------------------------------------------------------

    def _check_aggregates(self, items: Sequence[_Item], edges, roots: Set[str]) -> None:
        if len(self.sources) < 2:
            return
        for position, item in enumerate(items):
            if isinstance(item, _Group):
                if not _is_query_group(item):
                    self._check_aggregates(item, edges, roots)
                continue
            if item.upper not in _SUMMING_AGGREGATES:
                continue
            argument = items[position + 1] if position + 1 < len(items) else None
            if not isinstance(argument, _Group):
                continue
            if argument and _word(argument[0]) == "DISTINCT":
                continue
            refs = [self._resolve(q, c) for q, c in self._refs(argument)]
            touched = {alias for alias in refs if alias}
            if not touched and refs:
                continue  # only outer or unknown columns: can't judge
            call = f"{item.upper}({self._render(argument)})"
            if not roots:
                self._report(call, sorted(touched)[0] if touched else None, edges)
            elif touched and not touched & roots:
                self._report(call, sorted(touched)[0], edges)

    def _report(self, call: str, alias: Optional[str], edges) -> None:
        advice = (
            "Total each table separately in its own subquery, grouped by the shared "
            "key, then join those results. For counts, COUNT(DISTINCT ...) also works."
        )
        if alias is None:
            other = self._multiplier(None, edges)
            self.problems.append(
                f"Join fan-out: {call} counts rows after joining {other}, which repeats "
                f"rows of the other tables, so the count is inflated. {advice}"
            )
            return
        table = self.sources[alias].label
        other = self._multiplier(alias, edges)
        self.problems.append(
            f"Join fan-out: {call} uses values from {table}, but the join to {other} "
            f"repeats each {table} row once per matching {other} row, so the result is "
            f"inflated (or, for AVG, skewed). {advice}"
        )

    @staticmethod
    def _render(items: Sequence[_Item]) -> str:
        out = []
        for item in items:
            out.append(f"({_Scope._render(item)})" if isinstance(item, _Group) else item.text)
        text_ = " ".join(out).replace(" . ", ".").replace(" ,", ",")
        return text_ if len(text_) <= 60 else text_[:57] + "..."

    # -- output shape -------------------------------------------------------

    def _output(self, select_items: List[_Item], group_items: List[_Item], distinct: bool) -> _Source:
        names: List[Optional[str]] = []
        columns: Set[str] = set()
        for part in _split(select_items, ","):
            if not part:
                continue
            last = part[-1]
            before = part[-2] if len(part) > 1 else None
            if isinstance(last, _Token) and last.text == "*":
                qualifier = part[0].name if len(part) == 3 and isinstance(part[0], _Token) else None
                for alias, source in self.sources.items():
                    if qualifier is None or alias == qualifier:
                        columns |= source.columns
                names.append(None)
                continue
            name = last.name if isinstance(last, _Token) and _word(last) not in _NOT_COLUMNS else None
            explicit = _word(before) == "AS"
            plain_column = before is None or (isinstance(before, _Token) and before.text == ".")
            implicit_alias = before is not None and not (
                isinstance(before, _Token) and before.kind == "symbol"
            )
            if name and (explicit or plain_column or implicit_alias):
                names.append(name)
                columns.add(name)
            else:
                names.append(None)

        unique: List[FrozenSet[str]] = []
        group_parts = [p for p in _split(group_items, ",") if p]
        if group_parts:
            keys = set()
            for part in group_parts:
                key = self._group_key(part, names)
                if key is None:
                    keys = set()
                    break
                keys.add(key)
            if keys:
                unique.append(frozenset(keys))
        elif self._has_aggregate(select_items):
            unique.extend(frozenset({c}) for c in columns)
        elif distinct and len([n for n in names if n]) == len(names):
            unique.append(frozenset(n for n in names if n))
        return _Source(label="subquery", columns=columns, unique_sets=unique)

    def _group_key(self, part: List[_Item], names: List[Optional[str]]) -> Optional[str]:
        if len(part) == 1 and isinstance(part[0], _Token) and part[0].kind == "number":
            index = int(part[0].text) - 1
            return names[index] if 0 <= index < len(names) else None
        refs = list(self._refs(part))
        if len(refs) == 1 and len(part) in (1, 3):
            column = refs[0][1]
            return column if column in names else None
        return None

    @staticmethod
    def _has_aggregate(items: Sequence[_Item]) -> bool:
        for position, item in enumerate(items):
            if isinstance(item, _Group):
                if not _is_query_group(item) and _Scope._has_aggregate(item):
                    return True
                continue
            following = items[position + 1] if position + 1 < len(items) else None
            if item.upper in _SUMMING_AGGREGATES | {"MIN", "MAX"} and isinstance(following, _Group):
                return True
        return False


def _check_fanout_heuristics(sql: str, schema: Optional[_Schema] = None) -> Optional[str]:
    """Checks a query for joins that silently multiply totals.

    Each SELECT (including subqueries and CTEs) is checked on its own: every
    SUM, AVG, COUNT or TOTAL must add up columns of a table whose rows each
    appear once after the joins. COUNT(DISTINCT ...), MIN and MAX are never
    affected and are not checked.

    Args:
        sql: The SQL string to inspect.
        schema: Table columns and primary keys. Read from the database when
            not given.

    Returns:
        A message naming the aggregate and the join that multiplies it, or
        ``None`` if no problem was found or the SQL couldn't be analysed.
    """
    items = _nest(_tokenize(sql))
    if not items:
        return None
    while items and isinstance(items[-1], _Token) and items[-1].text == ";":
        items = items[:-1]
    if not items or _word(items[0]) not in ("SELECT", "WITH"):
        return None
    problems: List[str] = []
    try:
        _Scope(_database_schema() if schema is None else schema, {}, problems).statement(items)
    except (_Unparsed, IndexError, ValueError) as reason:
        logger.info("Critic: fan-out check skipped part of the query (%s).", reason)
    return problems[0] if problems else None


# ---------------------------------------------------------------------------
# Safety and result checks
# ---------------------------------------------------------------------------


def _is_safe_select(sql: str) -> bool:
    """Checks the query is a single read-only SELECT statement.

    Keywords are matched as whole words outside string literals, so a value
    such as ``'Delete me'`` or the ``replace()`` function is not mistaken for
    a write.

    Args:
        sql: The SQL string to inspect.

    Returns:
        True if the query is a safe SELECT, False otherwise.
    """
    tokens = _tokenize(sql)
    while tokens and tokens[-1].text == ";":
        tokens.pop()
    if not tokens or tokens[0].upper not in ("SELECT", "WITH"):
        return False
    if any(token.text == ";" for token in tokens):
        return False
    for position, token in enumerate(tokens):
        if token.upper in _FORBIDDEN:
            following = tokens[position + 1] if position + 1 < len(tokens) else None
            if token.upper == "REPLACE" and following is not None and following.text == "(":
                continue
            return False
    return True


def _has_aggregation(sql: str) -> bool:
    """True if the SQL groups rows anywhere (GROUP BY or a summing aggregate)."""
    tokens = _tokenize(sql)
    for position, token in enumerate(tokens[:-1]):
        following = tokens[position + 1]
        if token.upper == "GROUP" and following.upper == "BY":
            return True
        if token.upper in _SUMMING_AGGREGATES and following.text == "(":
            return True
    return False


def _check_repeated_groups(sql: str, df: pd.DataFrame) -> Optional[str]:
    """Catches a grouped result that lists the same group more than once.

    A query that totals by region and category should return one row per
    region and category. If the labels repeat, the totals were taken at a
    finer level (for example per location) and then labelled with the
    coarser one, so each row is only part of the figure it appears to show.

    Args:
        sql: The SQL that produced the result.
        df: The result.

    Returns:
        A message describing the repeated group, or None.
    """
    if len(df) < 2 or not _has_aggregation(sql):
        return None
    measures = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    labels = [c for c in df.columns if c not in measures]
    if not measures or not labels:
        return None
    repeated = df[df.duplicated(subset=labels, keep=False)]
    if repeated.empty:
        return None
    first = repeated.iloc[0]
    example = ", ".join(f"{c} = {first[c]}" for c in labels)
    count = int((df[labels] == first[labels]).all(axis=1).sum())
    return (
        f"The result has more than one row for the same {', '.join(map(str, labels))} "
        f"(for example {example} appears {count} times), so each row holds only part "
        f"of its group's total. Group by exactly the columns that are shown, so each "
        f"combination appears once with its full total."
    )


@lru_cache(maxsize=1)
def _query_engine():
    """The engine queries run on. SQLite files are opened read-only."""
    if DATABASE_URL.startswith("sqlite"):
        uri = f"{Path(DATABASE_PATH).resolve().as_uri()}?mode=ro"
        return create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(uri, uri=True),
            poolclass=NullPool,
        )
    return create_engine(DATABASE_URL)


def execute_and_validate(state: AnalysisState) -> AnalysisState:
    """Validates (safety + fan-out check), executes, then checks the result.

    Order of checks:
      1. Safety — must be a single read-only SELECT.
      2. Fan-out — static check of every join (see
         :func:`_check_fanout_heuristics`).
      3. Execution — run against the database (read-only) and capture the result.
      4. Repeated groups — a grouped result must list each group once
         (see :func:`_check_repeated_groups`).

    A failure at any stage sets ``error_message`` so the graph's retry loop
    routes back to the Programmer.

    Args:
        state: Pipeline state. Requires ``sql_query``.

    Returns:
        On success: state with ``execution_result`` (DataFrame), ``row_count``,
        and ``error_message`` cleared to "". On any failure: state with
        ``error_message`` set and ``execution_result`` cleared to None.
    """
    sql = state["sql_query"]
    logger.info("Critic: validating and executing SQL.")

    # 1. Safety gate.
    if not _is_safe_select(sql):
        msg = ("Rejected: only single read-only SELECT statements are allowed. "
               "The generated query was not a safe SELECT.")
        logger.warning("Critic: %s", msg)
        return {**state, "error_message": msg, "execution_result": None}

    # 2. Static fan-out check.
    heuristic_warning = _check_fanout_heuristics(sql)
    if heuristic_warning:
        logger.warning("Critic: fan-out check flagged query — %s", heuristic_warning)
        return {**state, "error_message": heuristic_warning,
                "execution_result": None}

    # 3. Execute.
    try:
        with _query_engine().connect() as conn:
            df = pd.read_sql_query(text(sql), conn)
    except Exception as exc:  # noqa: BLE001 — capture DB errors for the retry loop
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("Critic: execution failed — %s", error)
        return {**state, "error_message": error, "execution_result": None}

    if len(df) > MAX_RESULT_ROWS:
        df = df.head(MAX_RESULT_ROWS)

    # 4. Result check.
    repeated = _check_repeated_groups(sql, df)
    if repeated:
        logger.warning("Critic: result check flagged query — %s", repeated)
        return {**state, "error_message": repeated, "execution_result": None}

    logger.info("Critic: query succeeded — %d row(s).", len(df))
    return {
        **state,
        "execution_result": df,
        "row_count": len(df),
        "error_message": "",
    }
