# AgentOS-Bench

内置 benchmark 共 170 条：Single Tool、Multi Tool、Multi-hop、Dependency DAG、Fault Recovery 各 30，Long Horizon 20。生成文件中每条均有 `synthetic: true`、工具序列、答案、步数、故障和模板标识。

Dev 120，Test 50。Test 仅由运行脚本读取，不用于路由权重调整。四组策略共享工具、任务、seed、步数定义和 Mock provider；默认运行三次。记录 success、step success、tool selection、recovery、调用次数、token、成本和墙钟延迟。

Mock token 是本地程序统计，provider 字段标为 `mock-v1-offline`；成本为真实的 0，而不是估算 API 费用。统计代码提供 Bootstrap 95% CI、paired bootstrap 与 McNemar。

