# Algorithm design

## Adaptive routing

```text
score = α relevance + β smoothed_success - γ normalized_cost - δ normalized_latency
smoothed_success = (successes + a) / (successes + failures + a + b)
```

默认 `a=b=1`，冷启动为 0.5，避免一次成败形成极端概率。epsilon-greedy 提供探索；预算先过滤不可选 Agent。统计只能在当前任务结束后更新，当前决策不能读取未来标签。

## Execution acceptance

工具参数先经过 Draft 2020-12 JSON Schema 校验，覆盖 required、类型、枚举、范围和 additionalProperties，而不只是 required/extra。工具返回后，Verifier 检查非空、required output keys 和 MCP/节点 output schema；“没有抛异常”不等于业务成功。

## Recovery

- Retry：超时、限流、HTTP 5xx 和传输协议瞬时错误，最多 2 次；使用指数退避与有界 jitter。
- Repair：Schema/参数错误，产生新参数后最多重试 1 次。
- Replan：工具能力、空结果或程序化验收/计划逻辑失效，保留成功节点后最多 1 次，并持久化 `version+1` 完整计划。

状态机使用独立预算且最终抛出带 `ErrorType` 和 recovery action 的明确异常，避免无限循环。`INVALID_SCHEMA` 在普通 ValueError 之前分类；RemoteProtocol/TransportError 归为 SERVER_ERROR；空结果归为 EMPTY_RESULT；中断的非幂等 operation 归为 AMBIGUOUS_OPERATION，不自动重放。

## Retrieval

BM25 使用词频、逆文档频率和文档长度归一化。Dense 的默认离线实现是确定性 hashing vector，仅是无需模型下载的基线。Hybrid 使用 RRF：`score(d)=Σ 1/(k+rank_i(d))`。正式 BGE/reranker 实验必须安装 `rag` extra 并使用公开 qrels。

