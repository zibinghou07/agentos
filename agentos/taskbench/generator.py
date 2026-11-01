"""Generate TaskBench tasks over the synthetic world with programmatically computed gold.

Splits are separated by entity (customers, products, incidents, categories, orders),
so nothing tuned on dev reuses a test entity. Every generated task is marked
``synthetic: true``.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from agentos.taskbench.schema import (
    DbCheck,
    FaultSpec,
    NumberCheck,
    TaskSpec,
    TextCheck,
)
from agentos.world import data

Q1 = ("2026-01-01", "2026-03-31")
Q4 = ("2025-10-01", "2025-12-31")
CATEGORY_SPLIT = {
    "dev": ("Platform", "Retrieval", "Integration"),
    "test": ("Observability", "Hardware", "Security"),
}


def _customers(split: str) -> list[data.Customer]:
    ids = range(1, 9) if split == "dev" else range(9, 17)
    return [data.customer(f"c{i}") for i in ids]


def _products(split: str) -> list[data.Product]:
    return [p for p in data.PRODUCTS if (int(p.id[1:]) % 2 == 1) == (split == "dev")]


def _incidents(split: str) -> list[data.Incident]:
    return list(data.INCIDENTS[:4] if split == "dev" else data.INCIDENTS[4:])


def _orders_in(customer_id: str, start: str, end: str) -> list[data.Order]:
    return [o for o in data.ORDERS if o.customer_id == customer_id and start <= o.order_date <= end]


def _revenue(category: str, start: str, end: str) -> tuple[float, int]:
    revenue, units = 0.0, 0
    for order in data.ORDERS:
        item = data.product(order.product_id)
        if item.category == category and start <= order.order_date <= end:
            revenue += order.quantity * item.price
            units += order.quantity
    return round(revenue, 2), units


def _lookup(split: str) -> list[TaskSpec]:
    tasks = []
    for item in _products(split):
        tasks.append(
            TaskSpec(
                task_id=f"lookup-{item.id}",
                family="lookup",
                split=split,
                goal=f"What is the warranty period in months of {item.name}, and what is its list price in USD?",
                checks=[
                    NumberCheck(label="warranty_months", value=item.warranty_months, abs_tol=0),
                    NumberCheck(label="price", value=item.price),
                ],
                synthetic=True,
                reference_steps=1,
            )
        )
    return tasks


def _parallel(split: str) -> list[TaskSpec]:
    tasks = []
    for buyer in _customers(split):
        count = len(_orders_in(buyer.id, *Q1))
        tasks.append(
            TaskSpec(
                task_id=f"parallel-{buyer.id}",
                family="parallel_cross_source",
                split=split,
                goal=(
                    f"For our customer {buyer.name}: what first-response time (in hours) does "
                    "their support tier guarantee, and how many orders did they place between "
                    f"{Q1[0]} and {Q1[1]} inclusive?"
                ),
                checks=[
                    NumberCheck(label="sla_hours", value=data.SLA_HOURS[buyer.tier], abs_tol=0),
                    NumberCheck(label="q1_orders", value=count, abs_tol=0),
                ],
                synthetic=True,
                reference_steps=3,
            )
        )
    return tasks


def _multi_hop(split: str) -> list[TaskSpec]:
    tasks = []
    for incident in _incidents(split):
        item = data.product(incident.product_id)
        units = data.INVENTORY[(incident.warehouse_id, item.id)]
        tasks.append(
            TaskSpec(
                task_id=f"multihop-{incident.id.lower()}",
                family="multi_hop",
                split=split,
                goal=(
                    f"Postmortem {incident.id} describes a product failure linked to one "
                    "warehouse. Which product was affected, and how many units of that product "
                    "are currently in stock at that warehouse?"
                ),
                checks=[
                    TextCheck(label="product", aliases=[item.name]),
                    NumberCheck(label="units", value=units, abs_tol=0),
                ],
                synthetic=True,
                reference_steps=3,
            )
        )
    return tasks


def _analytics(split: str) -> list[TaskSpec]:
    tasks = []
    for category in CATEGORY_SPLIT[split]:
        q1_revenue, q1_units = _revenue(category, *Q1)
        q4_revenue, q4_units = _revenue(category, *Q4)
        if q4_revenue:
            change = (q1_revenue - q4_revenue) / q4_revenue * 100
            tasks.append(
                TaskSpec(
                    task_id=f"analytics-revenue-{category.lower()}",
                    family="analytics",
                    split=split,
                    goal=(
                        f"What was the total revenue of the {category} product category from "
                        f"{Q1[0]} to {Q1[1]}, and what is the percentage change compared with "
                        f"{Q4[0]} to {Q4[1]}?"
                    ),
                    checks=[
                        NumberCheck(label="q1_revenue", value=q1_revenue),
                        NumberCheck(label="pct_change", value=round(change, 2), rel_tol=0.01, abs_tol=0.2),
                    ],
                    synthetic=True,
                    reference_steps=3,
                )
            )
        if q4_units:
            change = (q1_units - q4_units) / q4_units * 100
            tasks.append(
                TaskSpec(
                    task_id=f"analytics-units-{category.lower()}",
                    family="analytics",
                    split=split,
                    goal=(
                        f"How many units of {category} products were sold from {Q1[0]} to "
                        f"{Q1[1]}, and by what percentage did that change versus {Q4[0]} to "
                        f"{Q4[1]}?"
                    ),
                    checks=[
                        NumberCheck(label="q1_units", value=q1_units, abs_tol=0),
                        NumberCheck(label="pct_change", value=round(change, 2), rel_tol=0.01, abs_tol=0.2),
                    ],
                    synthetic=True,
                    reference_steps=3,
                )
            )
    return tasks


def _policy_action(split: str) -> list[TaskSpec]:
    """Refund decisions with a balanced mix of eligible and ineligible cases."""
    tasks = []
    customers = {c.id for c in _customers(split)}
    candidates = [o for o in data.ORDERS if o.customer_id in customers]
    eligible_target = True
    for order in candidates:
        if len(tasks) >= 10:
            break
        window = data.refund_window_days(order)
        returned = order.status == "returned"
        placed = date.fromisoformat(order.order_date)
        # Put "today" just inside or just outside the window so the policy must be read.
        offset = window - 5 if eligible_target else window + 6
        today = placed + timedelta(days=offset)
        eligible = (not returned) and (today - placed).days <= window
        if eligible != eligible_target and not returned:
            continue
        eligible_target = not eligible_target
        buyer, item = data.customer(order.customer_id), data.product(order.product_id)
        amount = round(order.quantity * item.price, 2)
        checks = [
            DbCheck(
                label="refund_count",
                sql="SELECT COUNT(*) FROM refunds WHERE order_id=?",
                params=[order.id],
                expected=1 if eligible else 0,
                tolerance=0,
            ),
            DbCheck(
                label="denied_ticket_count",
                sql="SELECT COUNT(*) FROM tickets WHERE order_id=? AND type='refund_denied'",
                params=[order.id],
                expected=0 if eligible else 1,
                tolerance=0,
            ),
        ]
        if eligible:
            checks.append(
                DbCheck(
                    label="refund_amount",
                    sql="SELECT COALESCE(SUM(amount),0) FROM refunds WHERE order_id=?",
                    params=[order.id],
                    expected=amount,
                )
            )
        tasks.append(
            TaskSpec(
                task_id=f"policy-{order.id}",
                family="policy_action",
                split=split,
                goal=(
                    f"Today is {today.isoformat()}. Customer {buyer.name} requests a refund for "
                    f"order {order.id} ({item.name}). Apply our refund policy: if the request is "
                    "inside the refund window and the order was not already returned, issue a "
                    "full refund; otherwise open a refund_denied support ticket for the customer "
                    "referencing the order. Report what you did."
                ),
                checks=checks,
                synthetic=True,
                source={"eligible": eligible, "window_days": window, "returned": returned},
                reference_steps=4,
            )
        )
    return tasks


def _logistics(split: str) -> list[TaskSpec]:
    tasks = []
    buyers = _customers(split)
    products = _products(split)
    for index, buyer in enumerate(buyers):
        item = products[index % len(products)]
        hardware = item.category == "Hardware"
        days = data.SHIPPING_DAYS[buyer.region] + (1 if hardware else 0)
        total = sum(units for (w, p), units in data.INVENTORY.items() if p == item.id)
        tasks.append(
            TaskSpec(
                task_id=f"logistics-{buyer.id}-{item.id}",
                family="logistics",
                split=split,
                goal=(
                    f"Customer {buyer.name} wants to order {item.name}. How many business days "
                    "will standard (non-express) delivery to the customer's region take under "
                    "our shipping policy, and how many units of the product are in stock across "
                    "all warehouses combined?"
                ),
                checks=[
                    NumberCheck(label="business_days", value=days, abs_tol=0),
                    NumberCheck(label="total_units", value=total, abs_tol=0),
                ],
                synthetic=True,
                reference_steps=4,
            )
        )
    return tasks


def _faults(base: list[TaskSpec]) -> list[TaskSpec]:
    """Fault-injected variants of solvable tasks; the gold answer is unchanged."""
    plans = {
        "parallel_cross_source": [("fault_transient", FaultSpec(tool="list_orders", mode="transient"))],
        "multi_hop": [
            ("fault_outage", FaultSpec(tool="search_documents", mode="outage")),
            ("fault_transient", FaultSpec(tool="get_inventory", mode="transient")),
        ],
        "analytics": [("fault_outage", FaultSpec(tool="percent_change", mode="outage"))],
        "logistics": [("fault_transient", FaultSpec(tool="estimate_shipping", mode="transient"))],
    }
    variants = []
    for task in base:
        for family, fault in plans.get(task.family, []):
            variants.append(
                task.model_copy(
                    update={
                        "task_id": f"{task.task_id}-{fault.mode}-{fault.tool}",
                        "family": family,
                        "fault": fault,
                        "source": {**task.source, "base_task": task.task_id},
                    }
                )
            )
    return variants


def generate_tasks(split: str) -> list[TaskSpec]:
    base = (
        _lookup(split)
        + _parallel(split)
        + _multi_hop(split)
        + _analytics(split)
        + _policy_action(split)
        + _logistics(split)
    )
    return base + _faults(base)


def write_taskbench(directory: str | Path) -> dict[str, int]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("dev", "test"):
        tasks = generate_tasks(split)
        with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(task.model_dump_json() + "\n")
        counts[split] = len(tasks)
    (directory / "METADATA.json").write_text(
        json.dumps(
            {
                "name": "AgentOS-TaskBench Synthetic v1",
                "synthetic": True,
                "description": (
                    "End-to-end tasks over a synthetic enterprise world. Gold answers and "
                    "side-effect assertions are computed from agentos.world.data."
                ),
                "counts": counts,
                "splits": "entity-disjoint dev/test; dev for tuning, test locked",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return counts


def load_tasks(path: str | Path) -> list[TaskSpec]:
    with Path(path).open(encoding="utf-8") as handle:
        return [TaskSpec.model_validate_json(line) for line in handle if line.strip()]
