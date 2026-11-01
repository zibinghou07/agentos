# AgentOS-TaskBench：端到端任务级评测

旧的 `AgentOS-Bench`（`agentos/benchmarks/runner.py`）是一个**离线工具路由微基准**：它不调用 LLM，只比较几种"选哪个工具"的启发式规则。旧版里名为 `react` 的方法其实只是"在完整 Registry 里贪心选一个工具"，不是 ReAct。这个方法现已改名为 `greedy_top1_tool`，其余方法也改成了如实反映做法的名字：`keyword_rules`、`embedding_topk`、`clause_routing`。

TaskBench 评估的是**整个任务有没有做对**：最终答案里的数字和实体对不对，数据库里的副作用对不对。它不看某个节点选没选对工具。

## 1. 场景世界（`agentos/world/`，synthetic）

| 资源 | 规模 |
|---|---|
| 文档 | 40 篇：6 条政策（退款/SLA/物流/折扣/保修/数据保留）、12 份产品说明、8 份事故复盘、8 份 Runbook、6 篇干扰文档 |
| 数据库 | 12 个产品、16 个客户（gold/silver/bronze）、4 个仓库、48 条库存、120 条订单（固定种子生成），另有可写的 tickets、refunds 表 |
| MCP Server | 7 个：search、knowledge、catalog、sales、crm、ops、analytics |
| 工具 | 23 个；其中 `issue_refund`、`create_ticket` 通过 MCP `ToolAnnotations` 声明为**非幂等** |

工具返回的都是 Pydantic 模型，因此 MCP 会发布结构化的 output schema。Planner 用它为节点生成验收契约，Verifier 再按这个契约校验。

## 2. 任务族

每个 split 有 72 条 synthetic 任务，另外 test split 加入 25 条 HotpotQA。

| 族 | 每个 split | 考察能力 | 最少工具调用 |
|---|---|---|---|
| lookup | 6 | 单跳事实（保修期 + 价格） | 1 |
| parallel_cross_source | 8 | 客户 → 等级 → SLA 文档，同时查季度订单数（可并行） | 3 |
| multi_hop | 4 | 事故复盘 → 产品名 + 仓库 → 库存 | 3 |
| analytics | 6 | 两个季度营收/销量 + 百分比变化 | 3 |
| policy_action | 10 | 读退款政策 → 判断资格 → **退款或开 refund_denied 工单（副作用）** | 4 |
| logistics | 8 | 客户地区 + 硬件规则 → 配送天数；跨仓库库存求和 | 4 |
| fault_transient | 20 | 同上任务，关键工具前 2 次返回 503 | — |
| fault_outage | 10 | 同上任务，关键工具一直不可用，必须换工具或换计划 | — |
| hotpotqa_agentic | 25（test） | 真实 HotpotQA 问题；10 段上下文作为私有语料，**只能通过 MCP 搜索获取** | 2 |

**防泄漏：** dev 和 test 按实体隔离。dev 用客户 c1–c8、奇数编号产品、INC-201~204、Platform/Retrieval/Integration 类目；test 用其余实体。调参只用 dev，test 锁定不动。

## 3. 评分（全部由程序判定，不用模型打分）

- **数字检查**：从答案里抽取所有数字，只要有一个落在容差范围内（相对或绝对）就算通过。
- **文本检查**：规范化之后（小写、去标点、去冠词），按"连续 token 子串"匹配。
- **副作用检查**：运行结束后在该任务独立的 world DB 上执行 SQL。比如应退款的订单 `refunds` 必须恰好 1 条、金额必须正确；不应退款的订单必须 0 条退款、1 张 refund_denied 工单。
- **任务成功**：所有检查都通过。**score** 是通过检查的比例。HotpotQA 另外报告 token F1。

每个任务、每种方法都会新建一份 world，三种方法看到的是**完全相同**的工具、故障注入（`InstrumentedProvider`）和 LLM。

## 4. 方法

| 方法 | 说明 |
|---|---|
| `react` | 真正的 ReAct：单个 agent 看到全部 23 个工具，用原生 function calling 循环（Thought → Action → Observation），最多 12 轮 |
| `fixed_roles` | 固定角色顺序团队：researcher → data_analyst → support_agent 依次执行，各自只能用自己的工具，最后由 writer 汇总。没有 Planner、DAG、动态路由、验收和恢复 |
| `agentos` | 完整系统：Planner 生成多 agent 并行 DAG → 自适应路由 → 按 agent 限定工具范围 → 结构化选工具 → Retry/Repair → Verifier（按 action 本地重跑）→ Replan → Synthesizer |
| `agentos_static_routing` | 消融：只按 Planner 指定的 agent 执行，工具只按相关度排序，不读取运行统计 |
| `agentos_no_recovery` | 消融：去掉 retry、repair、选工具修复、节点重跑和 replan |
| `agentos_sequential` | 消融：并发度设为 1，用来观察 DAG 并行对延迟的贡献 |

## 5. 指标

任务成功率（bootstrap 95% CI）、平均 check score、LLM 调用次数、输入/输出/思考 token（取自 API 返回的真实 usage）、工具调用和工具错误次数、P50/P95 延迟、副作用违规数、计划节点数与并行度（`nodes/depth`）、replan 次数和节点重跑次数。

AgentOS 与每个基线之间做配对检验：McNemar 检验，以及成功率差值的配对 bootstrap CI。

## 6. 复现

```bash
make taskbench-dry          # 只看任务构成，不发请求
make taskbench-smoke        # dev 每族 1 条，小额冒烟
make taskbench              # test 全量：3 种方法 + 2 个消融
```

结果写入 `experiments/results/taskbench/<split>/runs.jsonl`（逐条追加，中断后可续跑）和 `summary.json`。脚本拒绝把 mock provider 的结果当作评测结果输出。

## 7. 结果

结果只在真实 API 运行后填写，没有运行过的数字一律不写。完整表格和失败分析见 [`experiment_report.md`](experiment_report.md)。

test 试点（`qwen3.8-27b`，n=18）：

| 方法 | 成功率 | 平均 token | P95 延迟 |
|---|---|---|---|
| react | 0.889 | 14216 | 52.8s |
| fixed_roles | 0.889 | 19743 | 92.9s |
| agentos | 0.556 | 8580 | 87.4s |

AgentOS 比 react 少 39.6% token，但成功率低 33.3 个百分点（McNemar p=0.077）；5/8 的失败来自最终答案的 JSON 格式问题。

修复后在**没用过的** 18 条 test 任务上重测（`qwen3.8-max-0902`）：

| 方法 | 成功率 | 平均 token | P95 延迟 |
|---|---|---|---|
| react | 0.944 | 14948 | 46.0s |
| fixed_roles | 0.667 | 16922 | 95.4s |
| agentos | 0.944 | 6853 | 53.9s |

AgentOS 与 react 成功率持平，token 少 54.2%；比 fixed_roles 成功率高 27.8 个百分点（McNemar p=0.13），token 少 59.5%。
