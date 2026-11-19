"""Direct unit test of the Critic's checks — no LLM involved.

Run from the project folder:  python test_heuristics.py

Feeds queries straight into the Critic's functions:

* the fan-out check, with the two real bugs from this project's debugging
  history, the correct queries that the old check wrongly rejected, and
  further bad and good patterns;
* the safety check (single read-only SELECT);
* the repeated-group check on query results.

The exit code is 0 only when every case passes.
"""

import sys

import pandas as pd

from src.agents.critic import (
    _check_fanout_heuristics,
    _check_repeated_groups,
    _database_schema,
    _is_safe_select,
)

# The demo database's tables: (columns, primary key). Passed in explicitly so
# the fan-out cases don't depend on reading the database.
SCHEMA = {
    "locations": (frozenset({"id", "name", "city", "region", "opened_date", "sq_footage"}), frozenset({"id"})),
    "services": (frozenset({"id", "name", "category", "base_price", "duration_minutes"}), frozenset({"id"})),
    "employees": (frozenset({"id", "location_id", "name", "role", "hourly_rate", "hire_date"}), frozenset({"id"})),
    "transactions": (frozenset({"id", "location_id", "service_id", "employee_id", "date", "quantity",
                                "unit_price", "discount", "total_revenue"}), frozenset({"id"})),
    "inventory": (frozenset({"id", "location_id", "product_name", "category", "quantity_on_hand",
                             "reorder_level", "unit_cost"}), frozenset({"id"})),
    "expenses": (frozenset({"id", "location_id", "category", "amount", "date", "notes"}), frozenset({"id"})),
}

# ---------------------------------------------------------------------------
# Fan-out: queries that must be flagged
# ---------------------------------------------------------------------------

# Bug #1 — plain COUNT() next to SUM() across a join (the first bug we hit).
BAD_PLAIN_COUNT = """
SELECT locations.id, locations.name,
       SUM(transactions.total_revenue) / COUNT(employees.id) AS revenue_per_employee
FROM employees
JOIN transactions ON employees.id = transactions.employee_id
JOIN locations ON transactions.location_id = locations.id
GROUP BY locations.id, locations.name
ORDER BY revenue_per_employee DESC
LIMIT 1;
"""

# Bug #2 — double join, COUNT(DISTINCT) fixed but cartesian fan-out remained
# (the second, worse bug — the one where the division silently cancelled out).
BAD_DOUBLE_JOIN = """
SELECT locations.id, locations.name,
       SUM(transactions.total_revenue) / COUNT(DISTINCT employees.id) AS revenue_per_employee
FROM locations
JOIN transactions ON locations.id = transactions.location_id
JOIN employees ON employees.location_id = locations.id
GROUP BY locations.id, locations.name
ORDER BY revenue_per_employee DESC
LIMIT 1;
"""

# Single-join fan-out — counting employees after joining their transactions.
BAD_SINGLE_JOIN = """
SELECT employees.location_id,
       SUM(transactions.total_revenue) / COUNT(employees.id) AS revenue_per_employee
FROM employees
JOIN transactions ON employees.id = transactions.employee_id
GROUP BY employees.location_id
ORDER BY revenue_per_employee DESC
LIMIT 1;
"""

# Profit done the naive way: revenue repeated per expense row and vice versa.
BAD_NAIVE_PROFIT = """
SELECT l.name, SUM(t.total_revenue) - SUM(x.amount) AS profit
FROM locations l
JOIN transactions t ON t.location_id = l.id
JOIN expenses x ON x.location_id = l.id
GROUP BY l.name;
"""

# A location's floor area added up once per transaction.
BAD_PARENT_COLUMN = """
SELECT l.region, SUM(l.sq_footage) AS total_area, SUM(t.total_revenue) AS revenue
FROM locations l JOIN transactions t ON t.location_id = l.id
GROUP BY l.region;
"""

# Hourly rates averaged per transaction instead of per employee.
BAD_PARENT_AVERAGE = """
SELECT e.location_id, AVG(e.hourly_rate) AS average_rate
FROM employees e JOIN transactions t ON t.employee_id = e.id
GROUP BY e.location_id;
"""

# COUNT(*) over two sibling tables.
BAD_SIBLING_COUNT = """
SELECT l.name, COUNT(*) AS n
FROM locations l
JOIN transactions t ON t.location_id = l.id
JOIN expenses x ON x.location_id = l.id
GROUP BY l.name;
"""

# Joining two child tables on their shared foreign key.
BAD_CHILD_TO_CHILD = """
SELECT t.location_id, SUM(t.total_revenue) AS revenue
FROM transactions t JOIN expenses x ON x.location_id = t.location_id
GROUP BY t.location_id;
"""

# A forgotten join condition (a cross product).
BAD_CROSS_PRODUCT = """
SELECT s.category, SUM(t.total_revenue) FROM transactions t, services s GROUP BY s.category;
"""

# The same mistake hidden inside a CTE.
BAD_INSIDE_CTE = """
WITH x AS (
  SELECT l.id, SUM(t.total_revenue) AS revenue, COUNT(e.id) AS staff
  FROM locations l
  JOIN transactions t ON t.location_id = l.id
  JOIN employees e ON e.location_id = l.id
  GROUP BY l.id)
SELECT * FROM x ORDER BY revenue DESC;
"""

# USING join between two child tables.
BAD_USING = """
SELECT location_id, SUM(total_revenue) AS revenue, SUM(amount) AS spend
FROM transactions JOIN expenses USING (location_id)
GROUP BY location_id;
"""

# ---------------------------------------------------------------------------
# Fan-out: correct queries that must NOT be flagged
# ---------------------------------------------------------------------------

# The subquery pattern from the Programmer's instructions.
GOOD_SUBQUERY = """
SELECT l.id, l.name, rev.total_revenue / emp.employee_count AS revenue_per_employee
FROM locations l
JOIN (SELECT location_id, SUM(total_revenue) AS total_revenue
      FROM transactions GROUP BY location_id) rev ON rev.location_id = l.id
JOIN (SELECT location_id, COUNT(DISTINCT id) AS employee_count
      FROM employees GROUP BY location_id) emp ON emp.location_id = l.id
ORDER BY revenue_per_employee DESC
LIMIT 1;
"""

# A plain, single-table aggregate with no join at all.
GOOD_NO_JOIN = """
SELECT region, SUM(total_revenue) AS total_revenue
FROM transactions
GROUP BY region;
"""

# Rejected by the old check (two joins), but each join goes to a parent table.
# This is the query from show_retries.py whose rejection led to a wrong answer.
GOOD_REGION_AND_CATEGORY = """
SELECT l.region, s.category, SUM(t.total_revenue) AS total_revenue
FROM transactions t
JOIN services s ON t.service_id = s.id
JOIN locations l ON t.location_id = l.id
GROUP BY l.region, s.category;
"""

# The Whitefield query shape the old check kept rejecting.
GOOD_WHITEFIELD = """
SELECT AVG(total) AS average_revenue_per_employee
FROM (SELECT e.id, SUM(t.total_revenue) AS total
      FROM employees e
      JOIN transactions t ON t.employee_id = e.id
      JOIN locations l ON e.location_id = l.id
      WHERE l.name = 'Whitefield Auto Spa'
      GROUP BY e.id) per_employee;
"""

# Rejected by the old check (plain COUNT next to SUM across a join).
GOOD_COUNT_AND_SUM = """
SELECT l.name, COUNT(*) AS transactions, SUM(t.total_revenue) AS revenue
FROM transactions t JOIN locations l ON t.location_id = l.id
GROUP BY l.name;
"""

# A per-transaction expression that uses a parent's column.
GOOD_ROW_EXPRESSION = """
SELECT s.category, SUM(t.quantity * s.base_price) AS list_value
FROM transactions t JOIN services s ON s.id = t.service_id
GROUP BY s.category;
"""

# Unqualified columns, resolved through the schema.
GOOD_UNQUALIFIED = """
SELECT region, SUM(total_revenue) FROM transactions
JOIN locations ON transactions.location_id = locations.id GROUP BY region;
"""

# The profit query the app produced (two pre-aggregated subqueries).
GOOD_PROFIT = """
SELECT loc.id, loc.name, (rev.total_revenue - exp.total_expenses) AS profit
FROM locations loc
JOIN (SELECT location_id, SUM(total_revenue) AS total_revenue
      FROM transactions GROUP BY location_id) rev ON rev.location_id = loc.id
JOIN (SELECT location_id, SUM(amount) AS total_expenses
      FROM expenses GROUP BY location_id) exp ON exp.location_id = loc.id
ORDER BY profit DESC
LIMIT 1;
"""

# Expenses by location and category.
GOOD_EXPENSE_BREAKDOWN = """
SELECT l.id, l.name, e.category, SUM(e.amount) AS total_expenses
FROM expenses e JOIN locations l ON e.location_id = l.id
WHERE e.date BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY l.id, l.name, e.category
ORDER BY total_expenses DESC;
"""

# Profit with CTEs.
GOOD_CTE_PROFIT = """
WITH rev AS (SELECT location_id, SUM(total_revenue) AS revenue FROM transactions GROUP BY location_id),
     spend AS (SELECT location_id, SUM(amount) AS expenses FROM expenses GROUP BY location_id)
SELECT l.name, rev.revenue - spend.expenses AS profit
FROM locations l
JOIN rev ON rev.location_id = l.id
JOIN spend ON spend.location_id = l.id
ORDER BY profit DESC;
"""

# Summing a pre-aggregated CTE by region.
GOOD_CTE_ROLLUP = """
WITH rev AS (SELECT location_id, SUM(total_revenue) AS revenue FROM transactions GROUP BY location_id)
SELECT l.region, SUM(rev.revenue) AS revenue
FROM rev JOIN locations l ON l.id = rev.location_id
GROUP BY l.region;
"""

# Old-style comma join with the condition in WHERE.
GOOD_COMMA_JOIN = """
SELECT l.region, SUM(t.total_revenue)
FROM transactions t, locations l
WHERE t.location_id = l.id AND t.date >= '2024-01-01'
GROUP BY l.region;
"""

# A chain of parents: transaction -> employee -> location.
GOOD_CHAIN = """
SELECT l.name, SUM(t.total_revenue) AS revenue
FROM locations l
JOIN employees e ON e.location_id = l.id
JOIN transactions t ON t.employee_id = e.id
GROUP BY l.name;
"""

# LEFT JOIN with a date range inside the ON clause.
GOOD_LEFT_JOIN = """
SELECT s.name, COALESCE(SUM(t.total_revenue), 0) AS revenue
FROM services s
LEFT JOIN transactions t ON t.service_id = s.id AND t.date BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY s.name;
"""

# Last 12 months, anchored on the data's latest date.
GOOD_LAST_12_MONTHS = """
SELECT s.id, s.name, SUM(t.total_revenue) AS total_revenue
FROM services s
JOIN (SELECT service_id, SUM(total_revenue) AS total_revenue FROM transactions
      WHERE date BETWEEN date('2026-09-18', '-12 months') AND date('2026-09-18')
      GROUP BY service_id) t ON t.service_id = s.id
GROUP BY s.id, s.name
ORDER BY total_revenue DESC
LIMIT 3;
"""

# A correlated scalar subquery.
GOOD_SCALAR_SUBQUERY = """
SELECT l.name,
       (SELECT SUM(t.total_revenue) FROM transactions t WHERE t.location_id = l.id) AS revenue
FROM locations l;
"""

# A ranking over an unnamed subquery.
GOOD_WINDOW = """
SELECT name, revenue, RANK() OVER (ORDER BY revenue DESC) AS position
FROM (SELECT l.name, SUM(t.total_revenue) AS revenue
      FROM transactions t JOIN locations l ON l.id = t.location_id
      GROUP BY l.name);
"""

# Two totals combined with UNION ALL.
GOOD_UNION = """
SELECT 'revenue' AS kind, SUM(total_revenue) AS amount FROM transactions
UNION ALL
SELECT 'expenses', SUM(amount) FROM expenses;
"""

# Average hourly rate per location.
GOOD_EMPLOYEE_AVERAGE = """
SELECT l.name, AVG(e.hourly_rate) FROM employees e JOIN locations l ON e.location_id = l.id GROUP BY l.name;
"""

FANOUT_CASES = [
    ("Bug #1 (plain COUNT + SUM + join)", BAD_PLAIN_COUNT, True),
    ("Bug #2 (double join, no subquery)", BAD_DOUBLE_JOIN, True),
    ("Single-join fan-out (count of the parent table)", BAD_SINGLE_JOIN, True),
    ("Naive profit (two child tables)", BAD_NAIVE_PROFIT, True),
    ("Parent column summed per child row", BAD_PARENT_COLUMN, True),
    ("Parent column averaged per child row", BAD_PARENT_AVERAGE, True),
    ("COUNT(*) over two sibling tables", BAD_SIBLING_COUNT, True),
    ("Child joined to child", BAD_CHILD_TO_CHILD, True),
    ("Missing join condition", BAD_CROSS_PRODUCT, True),
    ("Fan-out inside a CTE", BAD_INSIDE_CTE, True),
    ("USING join between child tables", BAD_USING, True),
    ("Good query (subquery pattern)", GOOD_SUBQUERY, False),
    ("Good query (no join at all)", GOOD_NO_JOIN, False),
    ("Good: region and category (old check rejected)", GOOD_REGION_AND_CATEGORY, False),
    ("Good: Whitefield per employee (old check rejected)", GOOD_WHITEFIELD, False),
    ("Good: COUNT(*) and SUM over one join (old check rejected)", GOOD_COUNT_AND_SUM, False),
    ("Good: per-row expression with a parent column", GOOD_ROW_EXPRESSION, False),
    ("Good: unqualified columns", GOOD_UNQUALIFIED, False),
    ("Good: profit from the app", GOOD_PROFIT, False),
    ("Good: expenses by location and category", GOOD_EXPENSE_BREAKDOWN, False),
    ("Good: profit with CTEs", GOOD_CTE_PROFIT, False),
    ("Good: CTE rolled up by region", GOOD_CTE_ROLLUP, False),
    ("Good: comma join", GOOD_COMMA_JOIN, False),
    ("Good: chain of parent tables", GOOD_CHAIN, False),
    ("Good: LEFT JOIN with a date range", GOOD_LEFT_JOIN, False),
    ("Good: last 12 months", GOOD_LAST_12_MONTHS, False),
    ("Good: correlated scalar subquery", GOOD_SCALAR_SUBQUERY, False),
    ("Good: window over an unnamed subquery", GOOD_WINDOW, False),
    ("Good: UNION ALL of two totals", GOOD_UNION, False),
    ("Good: average hourly rate by location", GOOD_EMPLOYEE_AVERAGE, False),
]

SAFETY_CASES = [
    ("Plain SELECT", "SELECT 1;", True),
    ("Lower-case select", "select name from services", True),
    ("CTE then SELECT", "WITH a AS (SELECT 1 AS x) SELECT x FROM a;", True),
    ("Word 'delete' inside a string", "SELECT * FROM services WHERE name = 'Delete me';", True),
    ("replace() string function", "SELECT replace(name, ' ', '_') FROM services;", True),
    ("Two statements", "SELECT 1; DROP TABLE services;", False),
    ("CTE followed by DELETE on new lines", "WITH a AS (SELECT 1)\nDELETE\nFROM transactions", False),
    ("PRAGMA", "PRAGMA table_info(services);", False),
    ("UPDATE", "UPDATE services SET base_price = 0;", False),
]

_WRONG_GRAIN = pd.DataFrame({
    "region": ["South", "South", "South", "West"],
    "category": ["Wash", "Wash", "Detailing", "Wash"],
    "total_revenue": [77385.30, 149704.12, 348079.34, 51234.00],
})
_RIGHT_GRAIN = pd.DataFrame({
    "region": ["South", "South", "West"],
    "category": ["Wash", "Detailing", "Wash"],
    "total_revenue": [227089.42, 348079.34, 51234.00],
})
_RAW_ROWS = pd.DataFrame({
    "date": ["2024-03-01", "2024-03-01"], "total_revenue": [900.0, 450.0],
})

REPEATED_GROUP_CASES = [
    ("Totals at a finer level than their labels",
     "SELECT l.region, s.category, COALESCE(rev.total_revenue, 0) AS total_revenue FROM locations l "
     "JOIN services s ON 1=1 LEFT JOIN (SELECT location_id, service_id, SUM(total_revenue) AS total_revenue "
     "FROM transactions GROUP BY location_id, service_id) rev ON rev.location_id = l.id AND rev.service_id = s.id;",
     _WRONG_GRAIN, True),
    ("One row per group",
     "SELECT l.region, s.category, SUM(t.total_revenue) FROM transactions t JOIN services s ON s.id = t.service_id "
     "JOIN locations l ON l.id = t.location_id GROUP BY l.region, s.category;",
     _RIGHT_GRAIN, False),
    ("Raw rows, no grouping",
     "SELECT date, total_revenue FROM transactions ORDER BY date;",
     _RAW_ROWS, False),
]


def main() -> int:
    failures = 0

    print("Fan-out check")
    for label, sql, expect_flag in FANOUT_CASES:
        result = _check_fanout_heuristics(sql, SCHEMA)
        fired = result is not None
        ok = fired == expect_flag
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {'flagged' if fired else 'clean'}")
        if fired:
            print(f"         -> {result}")

    print("\nSafety check")
    for label, sql, expect_safe in SAFETY_CASES:
        safe = _is_safe_select(sql)
        ok = safe == expect_safe
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {'allowed' if safe else 'rejected'}")

    print("\nRepeated-group check")
    for label, sql, df, expect_flag in REPEATED_GROUP_CASES:
        result = _check_repeated_groups(sql, df)
        ok = (result is not None) == expect_flag
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {'flagged' if result else 'clean'}")
        if result:
            print(f"         -> {result}")

    print("\nSchema read from the database")
    schema = _database_schema()
    keys_ok = bool(schema) and all(schema.get(t, ((), ()))[1] == frozenset({"id"}) for t in SCHEMA)
    columns_ok = bool(schema) and all(schema.get(t, ((), ()))[0] == SCHEMA[t][0] for t in SCHEMA)
    ok = keys_ok and columns_ok
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {len(schema)} tables; columns and primary keys match the demo schema")

    total = len(FANOUT_CASES) + len(SAFETY_CASES) + len(REPEATED_GROUP_CASES) + 1
    print(f"\n{total - failures}/{total} passed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
