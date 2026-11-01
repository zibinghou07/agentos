# AgentOS-Real-200

该数据集把真实公开任务作为独立评测集接入，不替换原来的 170 条 Synthetic 压力集：Synthetic 适合可控故障、恢复与消融；Real-200 用于检查公开真实任务上的迁移表现。

## 构成

| 来源 | 数量 | 本地任务类型 | 许可证 |
|---|---:|---|---|
| BFCL V4 `live_multiple` | 100 | 多候选函数中的工具选择 | Apache-2.0 |
| HotpotQA dev distractor | 100 | 给定 10 个段落的多跳问答/证据检索 | CC BY-SA 4.0 |

每个来源按 `SHA-256(source_id)` 排序后取 100 条，再各自固定切分为 75 dev + 25 test，合计 150 dev + 50 test。BFCL 固定到仓库提交 `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`。HotpotQA 优先从官方 CMU 地址下载；若该旧站点发生 TLS 故障，脚本回退到 Hugging Face 数据集镜像的前 100 条，并在元数据里记录实际来源。`test.jsonl` 不得参与路由规则、提示词或检索参数调优。每条记录保留原始 ID、下载 URL、原始文件 SHA-256、许可证、原始答案/支持事实及 `synthetic: false`。

```bash
make real-data
make real-benchmark
```

下载脚本只保留转换后的 200 条，不把 HotpotQA 的 44MB 原文件写入仓库。实际下载响应的哈希写入 `METADATA.json`，可以检测固定提交或镜像响应是否发生变化。

## 指标边界

默认本地基线不调用大模型，也不执行 BFCL 描述的第三方 API：

- BFCL：BM25 在候选工具 schema 上做 `tool_name_accuracy@1`；只测路由，不是 BFCL 官方 AST/参数评分。
- HotpotQA：BM25 在题目自带 10 个段落上做 `supporting_title_recall@2`；不是官方 Answer/Supporting Facts/Joint EM/F1。

因此这些结果可用于验证数据管线和本地路由检索，但不能冒充官方排行榜成绩。后续接入百炼模型时，应另存模型名、endpoint 模式、温度、时间、usage 和原始预测，并使用官方 evaluator 计算完整指标。

## 许可与归因

- BFCL 数据及格式说明见其官方 Gorilla 仓库，遵循 Apache-2.0。
- HotpotQA 数据遵循 CC BY-SA 4.0。若重新分发修改后的数据，应保留归因并遵循相同方式共享要求。

本仓库的转换格式不改变问题文本或 ground truth，只增加统一字段和确定性本地切分。
