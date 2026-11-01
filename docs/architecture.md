# Architecture

## Runtime

`AgentRuntimeService` 是生产应用编排层。API 只接收 goal；Service 先通过配置的 LLM 生成最小 `PlanDraft`，再由 Runtime 补充 task identity、PENDING 状态、全局唯一 operation ID，并用 Pydantic `TaskPlan` 校验唯一节点、依赖存在性和无环性。模型不能决定运行状态或伪造成功结果。

Kahn 拓扑算法负责环检测；调度器每轮选择所有依赖均成功的节点，并在 semaphore 限制下并发。失败只跳过其后代，不相关分支继续。每个 ready 节点依次经过 Adaptive Agent Router、AgentExecutor、动态 MCP `list_tools`、结构化 Tool Decision、Draft 2020-12 JSON Schema 参数校验、MCP `call_tool` 和 Verifier。只有程序化验收通过才写 SUCCESS。

每次创建、计划、节点开始、验收、成功/失败、Replan、取消和任务终止都写 SQLite event stream。Checkpoint 保存可重建的各版本计划；tasks 表保存终态、取消标志和 runner lease；operation ledger 以稳定 `operation_id` 复用已确认成功结果。进程启动时恢复非终态任务，SQLite 租约防止同一 task 出现两个 runner。

operation ledger 不是外部系统的事务协调器。若进程在外部副作用完成后、SUCCESS 落库前崩溃，本地只能看到 RUNNING：幂等节点允许恢复重试，非幂等节点标记 `AMBIGUOUS_OPERATION` 并要求人工核对。因此语义是“成功结果去重 + 受控 at-least-once”，不是严格 Exactly-once。

## Replan

Retry 和 Repair 预算耗尽、工具能力不足、空结果或程序化验收失败时，Service 可进入一次有界 Replan。Planner 必须返回完整的新 PlanDraft；Runtime 保留已成功节点及结果，为新节点生成新的 operation ID，将 `TaskPlan.version` 加一并持久化，然后重新进入调度。任意普通函数返回值不能冒充新计划。

## API state

- `GET /api/tasks/{id}`：持久化状态、最新计划版本、失败原因和最终结果。
- `GET /api/tasks/{id}/trace`：按 event_id 排序的真实事件流。
- `GET /api/tasks/{id}/operations`：operation 状态、结果/错误和交付语义说明。
- `POST /resume`：通过进程内 job guard 与 SQLite lease 双重去重。
- `POST /cancel`：先持久化取消请求，Scheduler 在安全边界转为 CANCELLED。

## Framework comparison

- LangGraph 的核心抽象是 State、Node、Edge 与 Checkpointer，适合显式状态图和条件流；AgentOS 采用相同的显式状态思路，但核心 DAG 调度器保持小而可测。`examples/langgraph_demo.py` 是最小固定角色图。
- AgentScope 强调 Message/Formatter、Agent、Model、Tool、Memory，以及 ReAct 的 reasoning/acting 两阶段。AgentScope 1.0.21 仍导入 MCP v1 的 `streamablehttp_client`，与主环境 MCP v2.3.0 不兼容，因此其对照使用 `examples/agentscope-requirements.txt` 建独立环境；AgentOS 不维护第二套 Runtime，也不在主 Runtime 混用 v1/v2。
- MCP 中 host 持有 client，每个 server 暴露 tool/resource/prompt。AgentOS Server 用 SDK v2 的 `MCPServer`，测试以 `Client(server)` 经过协议层连接。

## Safety

数据库只暴露参数化、预定义的只读查询；Registry 拒绝未知参数，限制调用超时并记录统计。Agent 无任意 shell/Python/SQL 工具。

