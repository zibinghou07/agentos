# 阿里云百炼 API 配置

AgentOS 调用的是百炼 OpenAI-compatible Chat Completions，而不是“创建托管 Agent”接口。这样 Planner、Executor 和 Verifier 可以通过同一个 `BaseLLM` 切换模型。

## Token Plan（截图所示套餐）

1. 在百炼套餐概览中创建/复制完整 API Key。Token Plan Key 通常以 `sk-sp-` 开头，而且只在创建或重置时完整显示。
2. 在项目根目录新建 `.env`：

```dotenv
AGENTOS_PROVIDER=bailian
AGENTOS_BAILIAN_MODE=token_plan
AGENTOS_MODEL=qwen3.8-flash
DASHSCOPE_API_KEY=sk-sp-你的完整密钥
```

不要填写 `DASHSCOPE_BASE_URL`；程序会使用：

```text
https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

截图中的模型可以通过修改 `AGENTOS_MODEL` 切换，例如：

```text
qwen3.8-flash
qwen3.8-27b
qwen3.8-max-0902
kimi-k3
deepseek-v4.1-flash
deepseek-v4.1-flash-1017
```

具体可用性仍以控制台套餐状态为准。

## 普通百炼 API Key（含新人免费额度）

普通 Key 通常以 `sk-` 开头。新人免费额度也使用这种 Key，平台会在产生按量费用前自动优先抵扣免费额度：

```dotenv
AGENTOS_PROVIDER=bailian
AGENTOS_BAILIAN_MODE=pay_as_you_go
AGENTOS_MODEL=qwen3.8-flash
DASHSCOPE_API_KEY=sk-你的完整密钥
AGENTOS_BAILIAN_REGION=cn-beijing
AGENTOS_BAILIAN_WORKSPACE_ID=你的WorkspaceId
```

程序会构造 Workspace 专属地址。北京默认空间未配置 Workspace ID 时，可以回退到旧的 `dashscope.aliyuncs.com` 兼容地址；新项目优先填写 Workspace ID。即使 `.env` 遗留了 `token_plan`，程序也会根据普通 `sk-` Key 自动选择这一模式。

## 检查

仅显示脱敏配置，不调用模型：

```bash
.venv/bin/python scripts/check_bailian.py --dry-run
```

发送一次最多 32 output tokens 的连通性请求：

```bash
make api-check
```

成功后可使用统一 HTTP 接口：

```bash
make api
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好，请介绍 AgentOS"}]}'
```

响应包含模型内容以及服务端返回的 input/output token usage。没有 usage 时，代码会明确标记 `usage_estimated=true`。

`AGENTOS_TIMEOUT_SECONDS` 控制普通聊天调用；Planner 和结构化工具决策属于非流式编排请求，`AgentRuntimeService` 会使用至少 120 秒超时，避免把正常的长规划误判为失败。

## 安全与排错

- `.env` 已被 `.gitignore` 排除；不要把 Key 发到聊天、提交到 Git 或写进测试。
- `sk-sp-` Token Plan Key 和普通 `sk-` Key 的 Base URL 不可混用；Provider 会根据 Key 自动选择 endpoint，并检查显式自定义 URL。
- 401/403：优先检查 Key 类型、Base URL、地域和 Workspace 是否匹配。
- `model_not_found`：检查套餐中模型是否已开启、模型 Code 是否完全一致。
- 429：降低 `AGENTOS_MAX_CONCURRENCY`，或等待额度/限流恢复。
- API 的 seed 不一定保证完全确定；正式实验应记录模型 Code、temperature、时间戳和原始 usage。

官方参考：

- https://help.aliyun.com/zh/model-studio/token-plan-team-quickstart
- https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions/
- https://help.aliyun.com/zh/model-studio/first-api-call-to-qwen
