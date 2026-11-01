from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentos.world import data

SCHEMA = """
CREATE TABLE IF NOT EXISTS products(id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
  price REAL NOT NULL, warranty_months INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS customers(id TEXT PRIMARY KEY, name TEXT NOT NULL, tier TEXT NOT NULL,
  region TEXT NOT NULL, joined TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS warehouses(id TEXT PRIMARY KEY, region TEXT NOT NULL, city TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS inventory(warehouse_id TEXT NOT NULL, product_id TEXT NOT NULL,
  units INTEGER NOT NULL, PRIMARY KEY(warehouse_id, product_id));
CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, product_id TEXT NOT NULL,
  quantity INTEGER NOT NULL, order_date TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tickets(id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT NOT NULL,
  order_id TEXT, type TEXT NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS refunds(id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL,
  amount REAL NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL);
"""


@dataclass
class World:
    """One isolated copy of the business world: a SQLite file plus a document corpus."""

    db_path: Path
    documents: dict[str, str] = field(default_factory=lambda: dict(data.DOCUMENTS))

    @classmethod
    def create(
        cls,
        db_path: str | Path,
        documents: dict[str, str] | None = None,
        *,
        reset: bool = True,
    ) -> "World":
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset and path.exists():
            path.unlink()
        world = cls(path, dict(documents) if documents is not None else dict(data.DOCUMENTS))
        world._seed()
        return world

    def connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        uri = f"file:{self.db_path}?mode=ro" if read_only else f"file:{self.db_path}"
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _seed(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.executemany(
                "INSERT OR IGNORE INTO products VALUES(?,?,?,?,?)",
                [(p.id, p.name, p.category, p.price, p.warranty_months) for p in data.PRODUCTS],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO customers VALUES(?,?,?,?,?)",
                [(c.id, c.name, c.tier, c.region, c.joined) for c in data.CUSTOMERS],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO warehouses VALUES(?,?,?)",
                [(w.id, w.region, w.city) for w in data.WAREHOUSES],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO inventory VALUES(?,?,?)",
                [(w, p, units) for (w, p), units in data.INVENTORY.items()],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO orders VALUES(?,?,?,?,?,?)",
                [
                    (o.id, o.customer_id, o.product_id, o.quantity, o.order_date, o.status)
                    for o in data.ORDERS
                ],
            )

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect(read_only=True) as conn:
            return [dict(row) for row in conn.execute(sql, params)]
