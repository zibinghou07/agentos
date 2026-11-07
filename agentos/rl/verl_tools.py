"""verl (0.9.x) tool wrappers around ``PoolSearcher``.

Imported only on the training machine. Each rollout passes its question id through
``extra_info.tools_kwargs[<tool>].create_kwargs.qid``; the tool searches that
question's pool and returns exactly the text the local environment would.

Tool config (configs/rl/tools.yaml):
    tools:
      - class_name: agentos.rl.verl_tools.SearchTool
        config: {type: native, tool_name: bm25_search, data_dir: data/rl, dense: BAAI/bge-small-en-v1.5}
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from agentos.rl.data import read_jsonl
from agentos.rl.tools import (
    TOOL_SCHEMAS,
    Corpus,
    EmbeddingStore,
    HashingEncoder,
    PoolSearcher,
    SentenceTransformerEncoder,
)


@lru_cache(maxsize=1)
def _resources(data_dir: str, dense: str) -> tuple[Corpus, dict[str, list[str]], EmbeddingStore]:
    directory = Path(data_dir)
    corpus = Corpus.load(directory / "corpus.jsonl")
    pools = {}
    for path in sorted(directory.glob("*.jsonl")):
        if path.name != "corpus.jsonl":
            pools.update({row["qid"]: row["pool"] for row in read_jsonl(path)})
    if dense == "hashing":
        embeddings = EmbeddingStore(HashingEncoder())
    else:
        # Query encoding runs on CPU inside the agent-loop workers; the GPU is busy with vLLM.
        embeddings = EmbeddingStore.load(SentenceTransformerEncoder(dense, device="cpu"), directory)
    return corpus, pools, embeddings


@lru_cache(maxsize=4096)
def _searcher(data_dir: str, dense: str, qid: str) -> PoolSearcher:
    corpus, pools, embeddings = _resources(data_dir, dense)
    return PoolSearcher(corpus, pools[qid], embeddings)


class SearchTool(BaseTool):
    """One class for all three tools; ``config.tool_name`` selects which."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        self.tool_name = config["tool_name"]
        self.data_dir = os.path.expanduser(config.get("data_dir", "data/rl"))
        self.dense = config.get("dense", "hashing")
        self._instances: dict[str, str] = {}
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == self.tool_name)
        return OpenAIFunctionToolSchema.model_validate(schema)

    async def create(self, instance_id: str | None = None, **kwargs: Any) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        self._instances[instance_id] = create_kwargs["qid"]
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: Any) -> tuple[ToolResponse, float, dict]:
        qid = self._instances[instance_id]
        result = _searcher(self.data_dir, self.dense, qid).call(self.tool_name, parameters)
        # Step reward stays 0: the episode reward is computed from the full transcript.
        return ToolResponse(text=result.text), 0.0, {"ok": result.ok, "error_kind": result.error_kind or ""}

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        self._instances.pop(instance_id, None)
