"""Deterministic enterprise-operations world used by the runtime and TaskBench.

Everything here is synthetic. Ground truth for benchmark tasks is computed from
these tables, never written by hand, so graders and tools cannot drift apart.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    category: str
    price: float
    warranty_months: int


@dataclass(frozen=True)
class Customer:
    id: str
    name: str
    tier: str
    region: str
    joined: str


@dataclass(frozen=True)
class Warehouse:
    id: str
    region: str
    city: str


@dataclass(frozen=True)
class Order:
    id: str
    customer_id: str
    product_id: str
    quantity: int
    order_date: str
    status: str


@dataclass(frozen=True)
class Incident:
    id: str
    product_id: str
    warehouse_id: str
    day: str
    root_cause: str
    mitigation: str


PRODUCTS: tuple[Product, ...] = (
    Product("p1", "AgentOS Pro", "Platform", 99.0, 12),
    Product("p2", "RAG Toolkit", "Retrieval", 49.0, 12),
    Product("p3", "MCP Adapter", "Integration", 29.0, 6),
    Product("p4", "Vector Vault", "Retrieval", 149.0, 24),
    Product("p5", "Trace Lens", "Observability", 79.0, 12),
    Product("p6", "Edge Runner", "Hardware", 399.0, 24),
    Product("p7", "Policy Guard", "Security", 129.0, 12),
    Product("p8", "Prompt Studio", "Platform", 59.0, 6),
    Product("p9", "Sensor Hub", "Hardware", 249.0, 18),
    Product("p10", "Ledger Sync", "Integration", 39.0, 6),
    Product("p11", "Insight Board", "Observability", 89.0, 12),
    Product("p12", "Key Shield", "Security", 69.0, 12),
)

WAREHOUSES: tuple[Warehouse, ...] = (
    Warehouse("W1", "north", "Beijing"),
    Warehouse("W2", "east", "Shanghai"),
    Warehouse("W3", "south", "Shenzhen"),
    Warehouse("W4", "west", "Chengdu"),
)

CUSTOMERS: tuple[Customer, ...] = (
    Customer("c1", "Aurora Logistics", "gold", "north", "2024-03-02"),
    Customer("c2", "Beacon Health", "silver", "east", "2024-05-19"),
    Customer("c3", "Cobalt Retail", "bronze", "south", "2024-07-08"),
    Customer("c4", "Delta Finance", "gold", "east", "2023-11-21"),
    Customer("c5", "Ember Studios", "silver", "west", "2025-01-14"),
    Customer("c6", "Fjord Energy", "bronze", "north", "2025-02-27"),
    Customer("c7", "Granite Labs", "gold", "south", "2024-09-30"),
    Customer("c8", "Harbor Foods", "silver", "west", "2024-12-03"),
    Customer("c9", "Iris Mobility", "gold", "west", "2023-08-16"),
    Customer("c10", "Juniper Media", "bronze", "east", "2025-04-05"),
    Customer("c11", "Kestrel Air", "silver", "north", "2024-06-11"),
    Customer("c12", "Lumen Schools", "gold", "east", "2024-01-29"),
    Customer("c13", "Meridian Bank", "bronze", "west", "2025-03-18"),
    Customer("c14", "Nimbus Games", "silver", "south", "2024-10-22"),
    Customer("c15", "Orchid Pharma", "gold", "north", "2023-12-09"),
    Customer("c16", "Pioneer Farms", "bronze", "south", "2025-05-26"),
)

INCIDENTS: tuple[Incident, ...] = (
    Incident("INC-201", "p9", "W3", "2026-01-14", "a firmware build without the watchdog patch", "rolled back firmware and re-flashed affected units"),
    Incident("INC-202", "p4", "W2", "2026-01-22", "an index compaction job that exhausted disk space", "added disk quota alerts and moved compaction off-peak"),
    Incident("INC-203", "p6", "W1", "2026-02-03", "a batch of power adapters with the wrong voltage rating", "recalled the adapter batch and shipped replacements"),
    Incident("INC-204", "p1", "W4", "2026-02-11", "a scheduler deadlock under heavy concurrent replanning", "added lease expiry and bounded the replan budget"),
    Incident("INC-205", "p7", "W2", "2026-02-19", "an expired signing certificate in the policy bundle", "automated certificate rotation 30 days before expiry"),
    Incident("INC-206", "p11", "W3", "2026-03-02", "a dashboard query that ignored time-zone offsets", "normalized all timestamps to UTC at ingestion"),
    Incident("INC-207", "p2", "W1", "2026-03-09", "a tokenizer upgrade that broke BM25 index compatibility", "pinned the tokenizer version and rebuilt indexes"),
    Incident("INC-208", "p12", "W4", "2026-03-17", "hardware security keys mislabeled during packing", "added barcode verification at the packing station"),
)

# Inventory is a fixed formula so it can be audited by hand.
INVENTORY: dict[tuple[str, str], int] = {
    (w.id, p.id): (int(p.id[1:]) * 7 + int(w.id[1:]) * 13) % 47 + 3
    for w in WAREHOUSES
    for p in PRODUCTS
}

ORDER_START = date(2025, 10, 1)
ORDER_END = date(2026, 3, 31)


def _generate_orders() -> tuple[Order, ...]:
    rng = random.Random(20260101)
    span = (ORDER_END - ORDER_START).days
    orders = []
    for index in range(1, 121):
        customer = CUSTOMERS[rng.randrange(len(CUSTOMERS))]
        product = PRODUCTS[rng.randrange(len(PRODUCTS))]
        quantity = rng.choice((1, 1, 2, 2, 3, 4, 5, 10, 12))
        day = ORDER_START + timedelta(days=rng.randrange(span + 1))
        status = rng.choice(("delivered",) * 7 + ("shipped", "returned"))
        orders.append(
            Order(f"o{index:03d}", customer.id, product.id, quantity, day.isoformat(), status)
        )
    return tuple(orders)


ORDERS: tuple[Order, ...] = _generate_orders()

# Business rules. Documents describe them in prose; graders use these values.
REFUND_DAYS_STANDARD = 30
REFUND_DAYS_HARDWARE = 14
REFUND_DAYS_GOLD_BONUS = 45  # gold customers, non-hardware only
SLA_HOURS = {"gold": 4, "silver": 12, "bronze": 48}
SHIPPING_DAYS = {"north": 2, "east": 2, "south": 3, "west": 4}
BULK_DISCOUNT_UNITS = 10
BULK_DISCOUNT_RATE = 0.08


def product(product_id: str) -> Product:
    return next(item for item in PRODUCTS if item.id == product_id)


def customer(customer_id: str) -> Customer:
    return next(item for item in CUSTOMERS if item.id == customer_id)


def refund_window_days(order: Order) -> int:
    item, buyer = product(order.product_id), customer(order.customer_id)
    if item.category == "Hardware":
        return REFUND_DAYS_HARDWARE
    if buyer.tier == "gold":
        return REFUND_DAYS_GOLD_BONUS
    return REFUND_DAYS_STANDARD


def _policy_documents() -> dict[str, str]:
    return {
        "policy_refund": (
            "Refund policy. Customers may request a full refund within "
            f"{REFUND_DAYS_STANDARD} days of the order date. Hardware products (Edge Runner, "
            f"Sensor Hub) have a shorter window of {REFUND_DAYS_HARDWARE} days. Gold-tier "
            f"customers receive an extended {REFUND_DAYS_GOLD_BONUS}-day window on all "
            "non-hardware products. Orders already marked returned cannot be refunded again. "
            "The refund amount equals quantity multiplied by the unit list price. Requests "
            "outside the window must be recorded as a refund_denied support ticket."
        ),
        "policy_sla": (
            "Support SLA. First-response targets depend on customer tier: gold customers "
            f"receive a first response within {SLA_HOURS['gold']} hours, silver within "
            f"{SLA_HOURS['silver']} hours, and bronze within {SLA_HOURS['bronze']} hours. "
            "Severity-one outages are escalated to the on-call engineer regardless of tier."
        ),
        "policy_shipping": (
            "Shipping policy. Standard delivery takes "
            f"{SHIPPING_DAYS['north']} business days to the north region, "
            f"{SHIPPING_DAYS['east']} to the east, {SHIPPING_DAYS['south']} to the south and "
            f"{SHIPPING_DAYS['west']} to the west. Express delivery removes one business day "
            "but never goes below one day. Hardware shipments add one business day for "
            "safety inspection."
        ),
        "policy_discount": (
            f"Volume discount policy. Orders of {BULK_DISCOUNT_UNITS} or more units of a single "
            f"product receive an {int(BULK_DISCOUNT_RATE * 100)} percent discount on the line "
            "total. Discounts are not combined with promotional codes."
        ),
        "policy_warranty": (
            "Warranty policy. Each product carries the warranty period listed in its product "
            "sheet, counted from the order date. Valid warranty claims are recorded as a "
            "warranty support ticket and receive a replacement unit, not a refund."
        ),
        "policy_data_retention": (
            "Data retention. Execution traces and checkpoints are retained for 90 days. "
            "Customer support tickets are retained for three years. Refund records are kept "
            "for seven years to satisfy audit requirements."
        ),
    }


def _runbook_documents() -> dict[str, str]:
    return {
        "doc_agent": "AgentOS uses DAG scheduling to run independent task nodes concurrently and persists checkpoints in SQLite.",
        "doc_recovery": "Retry handles transient failures, Repair corrects invalid arguments, and Replan replaces an infeasible plan.",
        "doc_rag": "Hybrid retrieval combines sparse BM25 and dense semantic retrieval using reciprocal rank fusion.",
        "doc_mcp": "Model Context Protocol standardizes tool discovery and invocation between a host client and tool servers.",
        "runbook_rate_limit": "Runbook: when an upstream model API returns HTTP 429, back off exponentially with jitter and lower concurrency before retrying.",
        "runbook_lease": "Runbook: a runner holds a lease on a task and renews it with heartbeats; an expired lease lets another runner resume from the latest checkpoint.",
        "runbook_idempotency": "Runbook: non-idempotent operations such as refunds and ticket creation must be recorded in the operation ledger so a resumed task never repeats a confirmed side effect.",
        "runbook_inventory_audit": "Runbook: inventory counts are reconciled nightly per warehouse; a discrepancy above five units opens an audit ticket for the warehouse lead.",
    }


def _product_documents() -> dict[str, str]:
    documents = {}
    blurbs = {
        "Platform": "orchestrates multi-step agent workflows",
        "Retrieval": "indexes enterprise documents for search",
        "Integration": "connects external systems through adapters",
        "Observability": "visualizes execution traces and metrics",
        "Hardware": "runs inference at the edge in factories and stores",
        "Security": "enforces access and signing policies",
    }
    for item in PRODUCTS:
        documents[f"sheet_{item.id}"] = (
            f"Product sheet: {item.name} (product id {item.id}) is a {item.category} product "
            f"that {blurbs[item.category]}. List price {item.price:.2f} USD per unit. "
            f"Warranty period: {item.warranty_months} months from the order date."
        )
    return documents


def _incident_documents() -> dict[str, str]:
    documents = {}
    for incident in INCIDENTS:
        item = product(incident.product_id)
        warehouse = next(w for w in WAREHOUSES if w.id == incident.warehouse_id)
        documents[f"postmortem_{incident.id.lower().replace('-', '_')}"] = (
            f"Postmortem {incident.id}. On {incident.day} customers reported failures of "
            f"{item.name} units shipped from warehouse {warehouse.id} in {warehouse.city}. "
            f"Root cause: {incident.root_cause}. Mitigation: {incident.mitigation}."
        )
    return documents


def _distractor_documents() -> dict[str, str]:
    return {
        "news_roadmap": "Roadmap note: the next release explores federated retrieval and a visual DAG editor; no pricing changes are planned this quarter.",
        "news_partner": "Partner update: three resellers in the east region completed certification for the MCP Adapter and Ledger Sync integrations.",
        "faq_accounts": "FAQ: account owners can invite teammates, rotate API keys, and export usage reports from the console settings page.",
        "faq_billing": "FAQ: invoices are issued monthly in USD; tax is calculated from the billing address and is not included in list prices.",
        "memo_holiday": "Memo: warehouses close for the Spring Festival week; orders placed during closure ship on the first business day afterwards.",
        "memo_security": "Memo: all staff must enable hardware keys for console access; Key Shield devices are issued by the IT desk.",
    }


def build_documents() -> dict[str, str]:
    return {
        **_runbook_documents(),
        **_policy_documents(),
        **_product_documents(),
        **_incident_documents(),
        **_distractor_documents(),
    }


DOCUMENTS: dict[str, str] = build_documents()
