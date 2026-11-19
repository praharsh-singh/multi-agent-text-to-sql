"""Builds and populates the demo SQLite database for the Text-to-SQL agent.

Creates a realistic auto-services business schema (an Indian multi-city car
wash / detailing chain) with six related tables and several years of synthetic
transaction data.  The schema is intentionally normalized with foreign keys so
the agents must reason about joins — the core challenge of enterprise
Text-to-SQL.

Run standalone::

    python data/seed_database.py

The output ``carwash.db`` can be swapped for a real client database later by
changing only ``DATABASE_URL`` in ``src/config.py``.

The data always ends on ``_DATA_END`` rather than on the day the script is
run, so every transaction and expense comes out identical on every run and
the figures quoted in the README stay true. Only the two Faker-generated
date columns (``locations.opened_date`` and ``employees.hire_date``) still
depend on the run date; no revenue or expense figure uses them.
"""

import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

from faker import Faker
from sqlalchemy import (
    Column, Date, Float, ForeignKey, Integer, MetaData, String, Table,
    create_engine, insert, text,
)

# Allow running this file directly (python data/seed_database.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATABASE_PATH, DATABASE_URL  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# Deterministic output across runs.
random.seed(42)
fake = Faker("en_IN")
Faker.seed(42)

# The last day of data. Transactions and expenses cover the three years up to
# this date, whatever day the script is run.
_DATA_END = date(2026, 9, 18)

# Expenses are generated from the monthly bases in _build_expenses and then
# multiplied by this factor in the database, which leaves each location
# modestly profitable (total expenses are about 0.85 times total revenue).
_EXPENSE_SCALE = 0.25

metadata = MetaData()

# ---------------------------------------------------------------------------
# Schema Definition
# ---------------------------------------------------------------------------

locations = Table(
    "locations", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("city", String, nullable=False),
    Column("region", String, nullable=False),
    Column("opened_date", Date, nullable=False),
    Column("sq_footage", Integer, nullable=False),
)

services = Table(
    "services", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("category", String, nullable=False),
    Column("base_price", Float, nullable=False),
    Column("duration_minutes", Integer, nullable=False),
)

employees = Table(
    "employees", metadata,
    Column("id", Integer, primary_key=True),
    Column("location_id", Integer, ForeignKey("locations.id"), nullable=False),
    Column("name", String, nullable=False),
    Column("role", String, nullable=False),
    Column("hourly_rate", Float, nullable=False),
    Column("hire_date", Date, nullable=False),
)

transactions = Table(
    "transactions", metadata,
    Column("id", Integer, primary_key=True),
    Column("location_id", Integer, ForeignKey("locations.id"), nullable=False),
    Column("service_id", Integer, ForeignKey("services.id"), nullable=False),
    Column("employee_id", Integer, ForeignKey("employees.id"), nullable=False),
    Column("date", Date, nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("unit_price", Float, nullable=False),
    Column("discount", Float, nullable=False),          # 0.0–0.25
    Column("total_revenue", Float, nullable=False),
)

inventory = Table(
    "inventory", metadata,
    Column("id", Integer, primary_key=True),
    Column("location_id", Integer, ForeignKey("locations.id"), nullable=False),
    Column("product_name", String, nullable=False),
    Column("category", String, nullable=False),
    Column("quantity_on_hand", Integer, nullable=False),
    Column("reorder_level", Integer, nullable=False),
    Column("unit_cost", Float, nullable=False),
)

expenses = Table(
    "expenses", metadata,
    Column("id", Integer, primary_key=True),
    Column("location_id", Integer, ForeignKey("locations.id"), nullable=False),
    Column("category", String, nullable=False),
    Column("amount", Float, nullable=False),
    Column("date", Date, nullable=False),
    Column("notes", String, nullable=True),
)

# ---------------------------------------------------------------------------
# Static Reference Data
# ---------------------------------------------------------------------------

_LOCATIONS = [
    ("Bengaluru Central", "Bengaluru", "South"),
    ("Whitefield Auto Spa", "Bengaluru", "South"),
    ("Chennai Marina", "Chennai", "South"),
    ("Hyderabad HiTech", "Hyderabad", "South"),
    ("Mumbai Andheri", "Mumbai", "West"),
    ("Pune Kothrud", "Pune", "West"),
    ("Delhi Connaught", "Delhi", "North"),
    ("Kolkata Salt Lake", "Kolkata", "East"),
]

_SERVICES = [
    ("Basic Wash", "Wash", 300.0, 20),
    ("Premium Wash", "Wash", 600.0, 35),
    ("Deluxe Wash", "Wash", 900.0, 50),
    ("Interior Detailing", "Detailing", 1500.0, 90),
    ("Exterior Detailing", "Detailing", 1800.0, 100),
    ("Full Detailing", "Detailing", 3200.0, 180),
    ("Ceramic Coating", "Detailing", 12000.0, 300),
    ("Oil Change", "Maintenance", 1200.0, 40),
    ("Tire Rotation", "Maintenance", 500.0, 30),
    ("Wax Polish", "Add-on", 800.0, 45),
    ("Engine Cleaning", "Maintenance", 1400.0, 60),
    ("Underbody Wash", "Add-on", 400.0, 25),
]

_ROLES = ["Manager", "Technician", "Detailer", "Receptionist"]

_INVENTORY_PRODUCTS = [
    ("Car Shampoo", "Cleaning"), ("Carnauba Wax", "Polish"),
    ("Microfiber Cloths", "Supplies"), ("Tire Polish", "Polish"),
    ("Glass Cleaner", "Cleaning"), ("Engine Degreaser", "Cleaning"),
    ("Ceramic Coating Kit", "Premium"), ("Air Fresheners", "Add-on"),
    ("Dashboard Polish", "Polish"), ("Wash Mitts", "Supplies"),
]

_EXPENSE_CATEGORIES = ["Rent", "Utilities", "Salaries", "Supplies",
                       "Marketing", "Maintenance"]


# ---------------------------------------------------------------------------
# Row Generators
# ---------------------------------------------------------------------------

def _build_locations() -> List[Dict]:
    """Returns location rows with realistic opening dates and sizes."""
    rows = []
    for i, (name, city, region) in enumerate(_LOCATIONS, start=1):
        rows.append({
            "id": i, "name": name, "city": city, "region": region,
            "opened_date": fake.date_between(start_date="-8y", end_date="-3y"),
            "sq_footage": random.choice([2500, 3200, 4000, 4800, 5500]),
        })
    return rows


def _build_services() -> List[Dict]:
    """Returns the fixed service catalogue."""
    return [
        {"id": i, "name": n, "category": c,
         "base_price": p, "duration_minutes": d}
        for i, (n, c, p, d) in enumerate(_SERVICES, start=1)
    ]


def _build_employees(location_rows: List[Dict]) -> List[Dict]:
    """Generates 5–8 employees per location, always including one Manager."""
    rows = []
    emp_id = 1
    for loc in location_rows:
        count = random.randint(5, 8)
        for j in range(count):
            role = "Manager" if j == 0 else random.choice(_ROLES[1:])
            rate = {"Manager": 350, "Technician": 180,
                    "Detailer": 220, "Receptionist": 150}[role]
            rows.append({
                "id": emp_id,
                "location_id": loc["id"],
                "name": fake.name(),
                "role": role,
                "hourly_rate": float(rate + random.randint(-20, 40)),
                "hire_date": fake.date_between(
                    start_date=loc["opened_date"], end_date="today"),
            })
            emp_id += 1
    return rows


def _build_transactions(
    location_rows: List[Dict],
    service_rows: List[Dict],
    employee_rows: List[Dict],
    n: int = 20000,
) -> List[Dict]:
    """Generates ``n`` sales transactions across the past 3 years.

    Each transaction references a valid location, an employee at that location,
    and a service, with a computed ``total_revenue`` after discount.
    """
    emp_by_location: Dict[int, List[int]] = {}
    for e in employee_rows:
        emp_by_location.setdefault(e["location_id"], []).append(e["id"])

    start = _DATA_END - timedelta(days=3 * 365)
    span_days = (_DATA_END - start).days
    rows = []

    for tid in range(1, n + 1):
        loc = random.choice(location_rows)
        svc = random.choice(service_rows)
        emp_id = random.choice(emp_by_location[loc["id"]])
        qty = random.choices([1, 2, 3], weights=[80, 15, 5])[0]
        # Unit price wobbles ±10% around the service base price.
        unit_price = round(svc["base_price"] * random.uniform(0.9, 1.1), 2)
        discount = random.choices(
            [0.0, 0.05, 0.10, 0.15, 0.25],
            weights=[60, 15, 12, 8, 5])[0]
        total = round(qty * unit_price * (1 - discount), 2)
        rows.append({
            "id": tid,
            "location_id": loc["id"],
            "service_id": svc["id"],
            "employee_id": emp_id,
            "date": start + timedelta(days=random.randint(0, span_days)),
            "quantity": qty,
            "unit_price": unit_price,
            "discount": discount,
            "total_revenue": total,
        })
    return rows


def _build_inventory(location_rows: List[Dict]) -> List[Dict]:
    """Stocks every location with the full product catalogue."""
    rows = []
    inv_id = 1
    for loc in location_rows:
        for product, category in _INVENTORY_PRODUCTS:
            qty = random.randint(0, 120)
            rows.append({
                "id": inv_id,
                "location_id": loc["id"],
                "product_name": product,
                "category": category,
                "quantity_on_hand": qty,
                "reorder_level": random.choice([20, 30, 40]),
                "unit_cost": round(random.uniform(80, 2500), 2),
            })
            inv_id += 1
    return rows


def _build_expenses(location_rows: List[Dict]) -> List[Dict]:
    """Generates one expense row per location, category, and month (3 years).

    Amounts are returned unscaled; ``main`` applies ``_EXPENSE_SCALE`` in the
    database after inserting them.
    """
    rows = []
    exp_id = 1
    start = _DATA_END.replace(day=1) - timedelta(days=3 * 365)
    for loc in location_rows:
        month_cursor = start
        while month_cursor < _DATA_END:
            for cat in _EXPENSE_CATEGORIES:
                base = {"Rent": 120000, "Utilities": 35000, "Salaries": 280000,
                        "Supplies": 45000, "Marketing": 30000,
                        "Maintenance": 25000}[cat]
                rows.append({
                    "id": exp_id,
                    "location_id": loc["id"],
                    "category": cat,
                    "amount": round(base * random.uniform(0.85, 1.2), 2),
                    "date": month_cursor,
                    "notes": None,
                })
                exp_id += 1
            # Advance ~one month.
            month_cursor += timedelta(days=30)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Creates the schema and bulk-inserts all synthetic data."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
        logger.info("Removed existing database.")

    engine = create_engine(DATABASE_URL)
    metadata.create_all(engine)
    logger.info("Schema created with 6 tables.")

    loc_rows = _build_locations()
    svc_rows = _build_services()
    emp_rows = _build_employees(loc_rows)
    txn_rows = _build_transactions(loc_rows, svc_rows, emp_rows)
    inv_rows = _build_inventory(loc_rows)
    exp_rows = _build_expenses(loc_rows)

    with engine.begin() as conn:
        conn.execute(insert(locations), loc_rows)
        conn.execute(insert(services), svc_rows)
        conn.execute(insert(employees), emp_rows)
        conn.execute(insert(transactions), txn_rows)
        conn.execute(insert(inventory), inv_rows)
        conn.execute(insert(expenses), exp_rows)
        # Scale in SQL, so amounts round exactly as SQLite rounds them.
        conn.execute(
            text("UPDATE expenses SET amount = ROUND(amount * :scale, 2)"),
            {"scale": _EXPENSE_SCALE},
        )

    logger.info("=" * 52)
    logger.info("  Database seeded successfully")
    logger.info("=" * 52)
    logger.info("  locations    : %d", len(loc_rows))
    logger.info("  services     : %d", len(svc_rows))
    logger.info("  employees    : %d", len(emp_rows))
    logger.info("  transactions : %d", len(txn_rows))
    logger.info("  inventory    : %d", len(inv_rows))
    logger.info("  expenses     : %d", len(exp_rows))
    logger.info("  Location     : %s", DATABASE_PATH)


if __name__ == "__main__":
    main()
