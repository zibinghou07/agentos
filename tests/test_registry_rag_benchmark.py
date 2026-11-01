from agentos.benchmarks.generator import generate_tasks
from agentos.mcp.catalog import build_local_registry
from agentos.rag import BM25Retriever, DenseRetriever, HybridRetriever
from agentos.rag.evaluator import evaluate_rankings


async def test_registry_validation_and_idempotency():
    registry = build_local_registry(":memory:")
    first = await registry.call("query_product", {"product_id": "p1"}, operation_id="same")
    second = await registry.call("query_product", {"product_id": "missing"}, operation_id="same")
    assert first == second and first["name"] == "AgentOS Pro"


def test_benchmark_has_170_and_locked_50_test():
    tasks = generate_tasks()
    assert len(tasks) == 170
    assert sum(task.split == "test" for task in tasks) == 50
    assert all(task.synthetic for task in tasks)
    dev_templates = {task.template_id for task in tasks if task.split == "dev"}
    test_templates = {task.template_id for task in tasks if task.split == "test"}
    assert dev_templates.isdisjoint(test_templates)


def test_retrieval_metrics():
    docs = {"a": "checkpoint recovery restart", "b": "hybrid retrieval rrf"}
    hybrid = HybridRetriever(BM25Retriever(docs), DenseRetriever(docs))
    rankings = {"q": [result.document_id for result in hybrid.search("checkpoint recovery", 2)]}
    metrics = evaluate_rankings(rankings, {"q": {"a"}}, 2)
    assert metrics["recall@2"] == 1 and metrics["mrr@2"] == 1
