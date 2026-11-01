"""MCP servers over one :class:`World`. Each factory returns a fresh server instance.

Tools return Pydantic models so MCP publishes structured output schemas; the
Planner grounds node contracts on them and the Verifier checks against them.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from agentos.rag import BM25Retriever, DenseRetriever, HybridRetriever
from agentos.world.store import World

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True)
SIDE_EFFECT = ToolAnnotations(read_only_hint=False, idempotent_hint=False, destructive_hint=False)


class SearchHit(BaseModel):
    document_id: str
    text: str
    score: float


class DocumentRecord(BaseModel):
    document_id: str
    text: str


class ProductRecord(BaseModel):
    id: str
    name: str
    category: str
    price: float
    warranty_months: int


class InventoryRecord(BaseModel):
    product_id: str
    warehouse_id: str
    units: int


class OrderRecord(BaseModel):
    id: str
    customer_id: str
    product_id: str
    quantity: int
    order_date: str
    status: str
    # quantity x unit price: the most a refund for this order may be.
    total: float


_ORDER_SELECT = (
    "SELECT o.*, o.quantity * (SELECT price FROM products p WHERE p.id = o.product_id) AS total "
    "FROM orders o"
)


class RevenueSummary(BaseModel):
    start_date: str
    end_date: str
    category: str
    orders: int
    units: int
    revenue: float


class CustomerRecord(BaseModel):
    id: str
    name: str
    tier: str
    region: str
    joined: str


class TicketRecord(BaseModel):
    id: int
    customer_id: str
    order_id: str | None
    type: str
    summary: str
    created_at: str


class WarehouseRecord(BaseModel):
    id: str
    region: str
    city: str


class ShippingEstimate(BaseModel):
    region: str
    express: bool
    hardware: bool
    business_days: int


class RefundRecord(BaseModel):
    id: int
    order_id: str
    amount: float
    reason: str
    created_at: str


class CalcResult(BaseModel):
    expression: str
    value: float


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hits(results) -> list[SearchHit]:
    return [SearchHit(document_id=r.document_id, text=r.text, score=round(r.score, 6)) for r in results]


def build_search_server(world: World) -> MCPServer:
    mcp = MCPServer("agentos-search")
    retriever = BM25Retriever(world.documents)

    @mcp.tool(annotations=READ_ONLY)
    def search_documents(query: str, top_k: int = 3) -> list[SearchHit]:
        """Keyword search over policies, product sheets, postmortems and runbooks."""
        return _hits(retriever.search(query, max(1, min(top_k, 10))))

    @mcp.tool(annotations=READ_ONLY)
    def get_document(document_id: str) -> DocumentRecord:
        """Read the full text of one document by its exact identifier."""
        if document_id not in world.documents:
            raise ToolError(f"document not found: {document_id}")
        return DocumentRecord(document_id=document_id, text=world.documents[document_id])

    @mcp.tool(annotations=READ_ONLY)
    def list_documents(prefix: str = "") -> list[str]:
        """List document identifiers, optionally filtered by prefix such as policy_ or postmortem_."""
        return sorted(key for key in world.documents if key.startswith(prefix))

    return mcp


def build_knowledge_server(world: World) -> MCPServer:
    mcp = MCPServer("agentos-knowledge")
    bm25 = BM25Retriever(world.documents)
    dense = DenseRetriever(world.documents)
    hybrid = HybridRetriever(bm25, dense)

    @mcp.tool(annotations=READ_ONLY)
    def bm25_search(query: str, top_k: int = 3) -> list[SearchHit]:
        """Sparse BM25 retrieval; best for exact names, identifiers and rare terms."""
        return _hits(bm25.search(query, max(1, min(top_k, 10))))

    @mcp.tool(annotations=READ_ONLY)
    def dense_search(query: str, top_k: int = 3) -> list[SearchHit]:
        """Offline hashing dense retrieval baseline; tolerant to paraphrase."""
        return _hits(dense.search(query, max(1, min(top_k, 10))))

    @mcp.tool(annotations=READ_ONLY)
    def hybrid_search(query: str, top_k: int = 3) -> list[SearchHit]:
        """Reciprocal-rank fusion of BM25 and dense retrieval for evidence gathering."""
        return _hits(hybrid.search(query, max(1, min(top_k, 10))))

    @mcp.tool(annotations=READ_ONLY)
    def rerank(query: str, documents: list[str]) -> list[SearchHit]:
        """Rerank supplied passages by lexical relevance to the query."""
        local = {str(index): text for index, text in enumerate(documents)}
        return _hits(BM25Retriever(local).search(query, len(local)))

    return mcp


def build_catalog_server(world: World) -> MCPServer:
    mcp = MCPServer("agentos-catalog")

    @mcp.tool(annotations=READ_ONLY)
    def get_product(product_id: str) -> ProductRecord:
        """Look up one product by exact identifier such as p4 (resolve names with search_products)."""
        rows = world.query("SELECT * FROM products WHERE id=?", (product_id,))
        if not rows:
            raise ToolError(f"product not found: {product_id}")
        return ProductRecord(**rows[0])

    @mcp.tool(annotations=READ_ONLY)
    def search_products(keyword: str) -> list[ProductRecord]:
        """Find products whose name contains a keyword, e.g. 'Sensor'."""
        rows = world.query("SELECT * FROM products WHERE name LIKE ? ORDER BY id", (f"%{keyword}%",))
        return [ProductRecord(**row) for row in rows]

    @mcp.tool(annotations=READ_ONLY)
    def list_products(category: str = "") -> list[ProductRecord]:
        """List products, optionally restricted to one category such as Hardware."""
        sql, params = "SELECT * FROM products", ()
        if category:
            sql, params = sql + " WHERE lower(category)=lower(?)", (category,)
        return [ProductRecord(**row) for row in world.query(sql + " ORDER BY id", params)]

    @mcp.tool(annotations=READ_ONLY)
    def get_inventory(product_id: str, warehouse_id: str = "") -> list[InventoryRecord]:
        """Units in stock per warehouse for a product id such as p9 (resolve names with search_products)."""
        sql = "SELECT product_id, warehouse_id, units FROM inventory WHERE product_id=?"
        params: tuple = (product_id,)
        if warehouse_id:
            sql, params = sql + " AND warehouse_id=?", (product_id, warehouse_id)
        rows = world.query(sql + " ORDER BY warehouse_id", params)
        if not rows:
            raise ToolError(f"inventory not found for product {product_id}")
        return [InventoryRecord(**row) for row in rows]

    return mcp


def build_sales_server(world: World) -> MCPServer:
    mcp = MCPServer("agentos-sales")

    @mcp.tool(annotations=READ_ONLY)
    def get_order(order_id: str) -> OrderRecord:
        """Look up one order by exact identifier such as o042."""
        rows = world.query(f"{_ORDER_SELECT} WHERE id=?", (order_id,))
        if not rows:
            raise ToolError(f"order not found: {order_id}")
        return OrderRecord(**rows[0])

    @mcp.tool(annotations=READ_ONLY)
    def list_orders(
        customer_id: str = "",
        product_id: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> list[OrderRecord]:
        """List orders filtered by customer id (c7), product id (p4) and inclusive ISO date range."""
        clauses, params = [], []
        for column, value in (("customer_id", customer_id), ("product_id", product_id)):
            if value:
                clauses.append(f"{column}=?")
                params.append(value)
        if start_date:
            clauses.append("order_date>=?")
            params.append(start_date)
        if end_date:
            clauses.append("order_date<=?")
            params.append(end_date)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = world.query(f"{_ORDER_SELECT}{where} ORDER BY order_date, id", tuple(params))
        return [OrderRecord(**row) for row in rows]

    @mcp.tool(annotations=READ_ONLY)
    def aggregate_revenue(start_date: str, end_date: str, category: str = "") -> RevenueSummary:
        """Revenue (quantity x list price) for orders in an inclusive ISO date range, optional category."""
        sql = (
            "SELECT COUNT(*) AS orders, COALESCE(SUM(o.quantity),0) AS units, "
            "COALESCE(SUM(o.quantity*p.price),0) AS revenue FROM orders o "
            "JOIN products p ON p.id=o.product_id WHERE o.order_date BETWEEN ? AND ?"
        )
        params: tuple = (start_date, end_date)
        if category:
            sql, params = sql + " AND lower(p.category)=lower(?)", (start_date, end_date, category)
        row = world.query(sql, params)[0]
        return RevenueSummary(
            start_date=start_date,
            end_date=end_date,
            category=category or "all",
            orders=row["orders"],
            units=row["units"],
            revenue=round(float(row["revenue"]), 2),
        )

    return mcp


def build_crm_server(world: World) -> MCPServer:
    mcp = MCPServer("agentos-crm")

    @mcp.tool(annotations=READ_ONLY)
    def get_customer(customer_id: str) -> CustomerRecord:
        """Look up one customer account by identifier such as c7."""
        rows = world.query("SELECT * FROM customers WHERE id=?", (customer_id,))
        if not rows:
            raise ToolError(f"customer not found: {customer_id}")
        return CustomerRecord(**rows[0])

    @mcp.tool(annotations=READ_ONLY)
    def find_customer(name: str) -> list[CustomerRecord]:
        """Find customer accounts whose company name contains the given text."""
        rows = world.query("SELECT * FROM customers WHERE name LIKE ? ORDER BY id", (f"%{name}%",))
        return [CustomerRecord(**row) for row in rows]

    @mcp.tool(annotations=READ_ONLY)
    def list_tickets(customer_id: str) -> list[TicketRecord]:
        """List support tickets opened for a customer."""
        rows = world.query("SELECT * FROM tickets WHERE customer_id=? ORDER BY id", (customer_id,))
        return [TicketRecord(**row) for row in rows]

    @mcp.tool(annotations=SIDE_EFFECT)
    def create_ticket(customer_id: str, type: str, summary: str, order_id: str = "") -> TicketRecord:
        """Open a support ticket (types: refund_denied, warranty, audit, general). Not idempotent."""
        if not world.query("SELECT id FROM customers WHERE id=?", (customer_id,)):
            raise ToolError(f"customer not found: {customer_id}")
        created = _now()
        with world.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO tickets(customer_id, order_id, type, summary, created_at) VALUES(?,?,?,?,?)",
                (customer_id, order_id or None, type, summary, created),
            )
            ticket_id = int(cursor.lastrowid)
        return TicketRecord(
            id=ticket_id, customer_id=customer_id, order_id=order_id or None,
            type=type, summary=summary, created_at=created,
        )

    return mcp


def build_ops_server(world: World) -> MCPServer:
    mcp = MCPServer("agentos-ops")

    @mcp.tool(annotations=READ_ONLY)
    def get_warehouse(warehouse_id: str) -> WarehouseRecord:
        """Look up a warehouse (W1-W4) and its region and city."""
        rows = world.query("SELECT * FROM warehouses WHERE id=?", (warehouse_id,))
        if not rows:
            raise ToolError(f"warehouse not found: {warehouse_id}")
        return WarehouseRecord(**rows[0])

    @mcp.tool(annotations=READ_ONLY)
    def estimate_shipping(region: str, express: bool = False, hardware: bool = False) -> ShippingEstimate:
        """Business days to deliver to a region (north, east, south, west) per shipping policy."""
        from agentos.world.data import SHIPPING_DAYS

        key = region.lower()
        if key not in SHIPPING_DAYS:
            raise ToolError(f"region not found: {region}")
        days = SHIPPING_DAYS[key] + (1 if hardware else 0)
        if express:
            days = max(1, days - 1)
        return ShippingEstimate(region=key, express=express, hardware=hardware, business_days=days)

    @mcp.tool(annotations=SIDE_EFFECT)
    def issue_refund(order_id: str, amount: float, reason: str) -> RefundRecord:
        """Issue a refund for an order. Not idempotent: each call creates a refund record."""
        rows = world.query(
            "SELECT o.quantity*p.price AS total, o.status FROM orders o JOIN products p "
            "ON p.id=o.product_id WHERE o.id=?",
            (order_id,),
        )
        if not rows:
            raise ToolError(f"order not found: {order_id}")
        if rows[0]["status"] == "returned":
            raise ToolError(f"order {order_id} was already returned; refund rejected")
        if amount <= 0 or amount > rows[0]["total"] + 1e-6:
            raise ToolError(f"invalid argument: amount must be in (0, {rows[0]['total']:.2f}]")
        created = _now()
        with world.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO refunds(order_id, amount, reason, created_at) VALUES(?,?,?,?)",
                (order_id, amount, reason, created),
            )
            refund_id = int(cursor.lastrowid)
        return RefundRecord(id=refund_id, order_id=order_id, amount=amount, reason=reason, created_at=created)

    return mcp


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ToolError("invalid argument: only numeric arithmetic is allowed")


def build_analytics_server(world: World) -> MCPServer:
    del world  # stateless
    mcp = MCPServer("agentos-analytics")

    @mcp.tool(annotations=READ_ONLY)
    def calculate(expression: str) -> CalcResult:
        """Evaluate a pure arithmetic expression such as (1200-950)/950*100."""
        try:
            value = _evaluate(ast.parse(expression, mode="eval"))
        except (SyntaxError, ZeroDivisionError) as exc:
            raise ToolError(f"invalid argument: {exc}") from exc
        return CalcResult(expression=expression, value=round(value, 6))

    @mcp.tool(annotations=READ_ONLY)
    def percent_change(old: float, new: float) -> CalcResult:
        """Percentage change from old to new, e.g. old=200 new=250 gives 25.0."""
        if old == 0:
            raise ToolError("invalid argument: old must be non-zero")
        return CalcResult(expression=f"({new}-{old})/{old}*100", value=round((new - old) / old * 100, 6))

    return mcp


SERVER_FACTORIES = {
    "search": build_search_server,
    "knowledge": build_knowledge_server,
    "catalog": build_catalog_server,
    "sales": build_sales_server,
    "crm": build_crm_server,
    "ops": build_ops_server,
    "analytics": build_analytics_server,
}


def build_mcp_servers(world: World, names: tuple[str, ...] | None = None) -> dict[str, MCPServer]:
    selected = names or tuple(SERVER_FACTORIES)
    return {name: SERVER_FACTORIES[name](world) for name in selected}
