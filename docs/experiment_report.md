# Experiment report

本文件只记录由脚本实际生成的结果。运行 `make benchmark && make evaluate` 后，原始记录位于 `experiments/results/benchmark/runs.csv`，汇总位于 `summary.json`，统计检验位于 `statistical_tests.json`。Dashboard 也只读取这些文件。

结果解释边界：这是合成工具环境 + Mock provider 的系统算法实验，可验证调度、选择和恢复逻辑；不能外推为某个真实 LLM 在 BFCL/AgentBench 上的能力，也不能与真实 API Token/费用混合。

RAG 小型 sanity set 由 `scripts/run_rag_experiment.py` 实跑，目标是验证评测和融合链路。样本量不足以宣称 Hybrid 或微调模型显著优于基线。

## 2026-10-31 local run

在 Apple M5 Pro / Python 3.11 / MockProvider 上执行 50 个 locked test、4 方法、3 次重复（共 600 条记录）。Task Success：Fixed Multi-Agent 0.64，Adaptive 0.46，Embedding 0.18，ReAct 0.18。Adaptive 在 27 个重复故障任务上的 Recovery Success 为 1.00，其他方法为 0。Mock 成本均为 0；P95 仅代表本地微型工具逻辑，不代表 API 延迟。

这个结果不支持“Adaptive 总体优于 Fixed”的假设。失败分析显示分句 + 词法检索对 test 模板中的新措辞不稳；下一步应在 dev 上训练/选择真正的 BGE 工具描述向量与权重，再一次性评测 test，而不是依据本次 test 回调规则。

轻量 CPU metric-learning smoke 使用 4 个合成 hard-negative 三元组和 4 个隔离测试查询，120 epochs 后 loss=0.125；Recall@1 保持 0.50、MRR@4 保持 0.6875。结果证明训练/保存/评测链路可运行，但没有效果提升，不能替代 BGE 微调结论。


## 2026-11-01 旧微基准方法改名

`agentos/benchmarks/runner.py` 是一个离线工具路由微基准，不调用 LLM。为了如实反映做法，其中的方法已改名：`react` → `greedy_top1_tool`、`fixed_multi_agent` → `keyword_rules`、`embedding_routing` → `embedding_topk`、`adaptive_routing` → `clause_routing`。`lexical_similarity` 现在会把下划线切开，所以 `make benchmark` 重跑后 clause_routing 的成功率从 0.46 变为 0.64。这组数字只反映路由启发式，**不是 agent 对比**。

## 2026-11-01 TaskBench dev 冒烟（真实 API，n=9，仅用于调试）

配置：百炼 `qwen3.8-flash`，`AGENTOS_ENABLE_THINKING=false`，temperature 0，dev split 每个任务族 1 条（共 9 条），1 次重复。原始记录见 `experiments/results/taskbench/dev_smoke*/`。

| 运行 | 方法 | 成功率 | 平均 token | 平均 LLM 调用 | 平均工具调用 | P50 / P95 延迟 |
|---|---|---|---|---|---|---|
| v1 | react | 0.889 | 15128 | 4.7 | 4.9 | 24.1s / 39.3s |
| v1 | fixed_roles | 0.778 | 24062 | 13.4 | 13.0 | 72.2s / 79.0s |
| v1 | agentos | 0.556 | 15379 | 5.4 | 3.8 | 32.4s / 90.4s |
| v2（修复后） | agentos | 1.000 | 8097 | 6.0 | 4.1 | 29.8s / 68.7s |

v1 中 AgentOS 失败的 4 条任务，原因分别是：

1. Synthesizer 拿到了结构化答案（一个 dict），schema 校验失败，结果答案为空；
2. Replan 把整份计划连同运行时字段原样回显，超出 max_tokens，JSON 被截断（2 条）；
3. LLM 自己编的 `required_output_keys` 和工具真实的输出契约（list）不一致，Verifier 误判失败；
4. 修复参数时看不到上游结果，没法把产品名换成产品 id。

修复都是**通用机制**，没有针对某个任务写特例：答案归一化；replan 只返回新节点并允许依赖已保留节点；验收键只采用工具真实契约里的字段；参数修复时带上依赖结果。

**解释边界：** v2 用的 9 条任务就是调试 v1 时用的那 9 条，所以 1.000 **不能**当作结论。这里只能说明改动后机制能跑通，不能和 v1 的基线直接比较。正式结论必须来自锁定的 test split 全量运行。

## 2026-11-01 TaskBench test 试点评测（真实 API，n=18，结论以此为准）

配置：百炼 `qwen3.8-27b`，`AGENTOS_ENABLE_THINKING=false`，temperature 0，test split 每个任务族 2 条（9 族，共 18 条，其中 HotpotQA 2 条为公开数据），1 次重复，三种方法按任务交替运行，token 上限 900K（实际约 766K，未触发截断）。原始记录：`experiments/results/taskbench/test_qwen3.8-27b/`。这组任务在运行前没有被查看或用来调参。

| 方法 | 成功率（95% CI） | 平均检查分 | 平均 token | 平均 LLM 调用 | 平均工具调用 | P50 / P95 延迟 | 副作用违规 |
|---|---|---|---|---|---|---|---|
| react（原生 function calling） | 0.889 (0.72–1.00) | 0.944 | 14216 | 4.6 | 4.4 | 23.0s / 52.8s | 0 |
| fixed_roles | 0.889 (0.72–1.00) | 0.944 | 19743 | 12.2 | 10.4 | 72.6s / 92.9s | 0 |
| agentos | 0.556 (0.33–0.78) | 0.620 | 8580 | 6.4 | 4.3 | 33.3s / 87.4s | 2 |

配对检验（AgentOS 对比基线）：

- vs react：成功率差 −0.333（bootstrap 95% CI −0.61 到 −0.06），McNemar p=0.077；token 为 react 的 0.60 倍（**少 39.6%**）。
- vs fixed_roles：成功率差 −0.333（CI −0.56 到 −0.11），McNemar p=0.041；token 为 0.43 倍（**少 56.5%**）。
- AgentOS 计划平均 3.4 个节点，平均并行度 1.39，平均 replan 0.33 次。

**结论：** 在这组试点上，AgentOS 省 token，但成功率**显著低于**两个基线，P95 延迟比 react 高 65.6%。不能写“成功率提升”。

失败分析（8 条失败）：

1. **5 条：运行时 SUCCESS，但最终答案为空。** Synthesizer 输出格式不合规：`evidence_nodes` 里放了 dict（2 条）、JSON 后面多了字符（2 条）、带 ```json 代码块并被截断（1 条）。这是答案组装这一层的鲁棒性问题，工具链本身已经拿到了正确数据。
2. **2 条：合法的空结果被当作失败。** 客户 Q1 订单数的正确答案是 0，`list_orders` 返回空列表，被判为 EMPTY_RESULT，replan 后仍失败。
3. **1 条：`calculate` 收到非数值表达式**，replan 后失败；另外 policy 族有 2 次副作用违规（多建或漏建 denied ticket）。
4. 观察到多个节点执行时被路由到 `general_executor`，而不是规划时的专业 agent，需要进一步排查。

**后续规则：** 这 18 条已经看过失败原因，以后不能再拿它们评估修复效果。修复只能用 dev 验证，然后在**没用过的** test 任务（例如每族第 3–4 条）上重新评测，并在报告里单独列出版本。

### 2026-11-01 针对 test 试点失败模式的通用修复（尚未重新评测）

- `extract_json` 只取第一个完整 JSON 值，兼容尾随字符和未闭合的 ```json 代码块。
- `FinalAnswer.evidence_nodes` 接受节点对象并取其 id；Synthesizer 解析失败时带错误提示重试 1 次，`max_tokens` 700 → 1000。
- 输出契约是列表的工具，空列表是合法答案（例如“0 个订单”），Recovery 和 Verifier 都不再把它当作 EMPTY_RESULT。
- `general_executor` 只做兜底：只有没有合格的专业 agent，或专业 agent 已有 ≥3 次记录且 Beta 成功率 < 0.5 时，它才参与路由。
- `scripts/run_taskbench.py --skip-per-family 2` 跳过已分析过的 18 条，保证重测用的是没看过的 test 任务（已确认与试点 0 重叠）。

这些修复的依据是错误类型，不是具体任务的答案。效果还没有测，不能引用。

## 2026-11-01 修复后 dev 检查（qwen3.8-max-0902，n=9，仅调试）

AgentOS 9 条 dev 任务通过 7 条，平均 7010 token（`experiments/results/taskbench/dev_max_v3/`）。其中退款金额错误的原因是订单记录没有总额字段，模型把数量 3 当成了金额（正确值 297）。为此给 `OrderRecord` 加了 `total`（数量 × 单价）。所有方法共用同一套工具，单条重跑后通过（`dev_max_v3b/`）。另一条故障任务是模型在 replan 后把事故编号当成产品 id 传入，属于模型推理错误，未做针对性修改。

## 2026-11-01 TaskBench test 新任务评测（qwen3.8-max-0902，n=18，修复后版本）

配置：test split 每族跳过已分析的前 2 条，取第 3–4 条（`--skip-per-family 2 --per-family 2`），与 27b 试点 0 重叠，其中 2 条 HotpotQA。`AGENTOS_ENABLE_THINKING=false`，temperature 0，1 次重复。本次评测共约 70 万 token（ReAct+AgentOS 约 39.2 万，fixed_roles 约 30.5 万）。原始记录：`experiments/results/taskbench/test_max_fresh/`。

| 方法 | 成功率（95% CI） | 平均检查分 | 平均 token | 平均 LLM 调用 | 平均工具调用 | P50 / P95 延迟 | 副作用违规 |
|---|---|---|---|---|---|---|---|
| react | 0.944 (0.83–1.00) | 0.963 | 14948 | 4.5 | 3.5 | 29.7s / 46.0s | 1 |
| fixed_roles | 0.667 (0.44–0.89) | 0.778 | 16922 | 10.1 | 6.3 | 68.0s / 95.4s | 0 |
| agentos | 0.944 (0.83–1.00) | 0.944 | 6853 | 5.4 | 3.7 | 30.4s / 53.9s | 1 |

配对检验：

- vs react：成功率差 0（CI −0.17 到 +0.17，McNemar p=0.48）；token 为 0.458 倍（**少 54.2%**）；P95 延迟高 17.2%。
- vs fixed_roles：成功率高 27.8 个百分点（bootstrap CI +0.05 到 +0.56；McNemar p=0.13，未达 0.05 显著）；token 少 59.5%；P95 延迟低 43.5%。
- AgentOS 平均并行度 1.49，平均 replan 0.06 次。

**解释边界：**

- 这次和 27b 试点的模型不同、任务不同，又隔着一次修复，所以**不能**写成“修复把成功率从 55.6% 提到 94.4%”。两次结果各自只说明本次之内的方法对比。
- n=18 是小样本：对 ReAct 只能说“成功率持平、token 减半”；对 fixed_roles 的成功率优势 bootstrap CI 不含 0，但 McNemar 不显著。
