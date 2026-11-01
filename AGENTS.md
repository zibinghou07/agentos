# AgentOS contributor guide

- Python 3.11+, typed public APIs, async for I/O and scheduling.
- Never commit API keys, generated databases, model weights, or fabricated metrics.
- Mark generated benchmark tasks with `synthetic: true`.
- Test recovery, idempotency, and failure branches whenever runtime behavior changes.
- Keep test-set data out of routing/retrieval tuning.

