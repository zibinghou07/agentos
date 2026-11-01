# Failure analysis

已覆盖故障：循环依赖、上游失败、超时、无效参数、空结果、重复 operation、主动取消。典型案例是 `query_product` 收到 `wrong` 字段：Registry 拒绝；Adaptive 方法的 Repair 产生 `product_id` 新参数后成功，其他策略在预算耗尽后失败。

尚未在本机验证：真实 API 429/网络分区、stdio 子进程崩溃、跨进程并发写冲突、CUDA OOM、公开数据集字段漂移。它们不能由 Mock 测试替代。

实际进程中断测试使用 `scripts/checkpoint_crash_demo.py`：节点 one、two 成功后，three 调用 `os._exit(73)`。新进程恢复日志为 `one,two,three,three`，证明成功节点没有重放；RUNNING 的 three 按 at-least-once 语义重试。外部副作用工具仍需 operation_id 才能收敛为等效一次。

