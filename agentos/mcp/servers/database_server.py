import sqlite3
from pathlib import Path

from mcp.server import MCPServer

from agentos.mcp.catalog import initialize_business_database

DB_PATH = Path("data/business.db")
initialize_business_database(DB_PATH)
mcp = MCPServer("agentos-database")


def _one(sql: str, params: tuple) -> dict | None:
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


@mcp.tool()
def query_product(product_id: str) -> dict | None:
    """Look up a product by exact identifier."""
    return _one("SELECT id,name,price FROM products WHERE id=?", (product_id,))


@mcp.tool()
def search_products(keyword: str) -> list[dict]:
    """Find products by keyword using parameterized read-only SQL."""
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT id,name,price FROM products WHERE name LIKE ?", (f"%{keyword}%",))]


@mcp.tool()
def query_order(order_id: str) -> dict | None:
    """Look up an order by exact identifier."""
    return _one("SELECT id,product_id,quantity,order_date FROM orders WHERE id=?", (order_id,))


@mcp.tool()
def aggregate_sales(start_date: str, end_date: str) -> dict:
    """Aggregate revenue between inclusive ISO dates."""
    return _one("SELECT COALESCE(SUM(o.quantity*p.price),0) AS revenue FROM orders o JOIN products p ON p.id=o.product_id WHERE o.order_date BETWEEN ? AND ?", (start_date, end_date)) or {"revenue": 0}


if __name__ == "__main__":
    mcp.run()

