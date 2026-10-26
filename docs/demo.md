# 演示用例

以下用例用于手动验证 Agent 的关键执行行为。回答措辞每次可能不同，判断是否通过以 `tool_trace`、`plan` 和 `status` 为准。

准备：在 `.env` 中配置模型 Key，并设置 `TOOL_TRANSPORT=mcp`；启动 `python -m uvicorn app.api:app --reload`，打开 `http://127.0.0.1:8000/docs`。每次全新演示请换一组 `thread_id`。

售后工单的多轮流程（草稿 → 确认提交 → 查询进度）见 README 的「PowerShell 调用示例」。

## 用例一：连续工具调用

目的：验证模型拿到上一步工具结果后，能继续发起下一次工具调用。

```json
{"message": "请先用计算工具计算23乘47，再用计算工具把结果乘以2，两步分别调用。", "thread_id": "demo-chain-101"}
```

预期：最终结果 2162；两条 `calculate` trace，第二次参数为 `a=1081`、`b=2`；`plan` 与 trace 的 `call_id` 一一对应。

说明：模型输出的只是工具调用请求，实际执行由 Python / MCP 完成。`ToolMessage` 返回后，图重新进入 LLM 节点，由模型决定下一步。

## 用例二：服务重启后的历史恢复

目的：区分进程内 checkpoint 与 SQLite 会话持久化。

第一轮：

```json
{"message": "请用计算工具计算23乘47。", "thread_id": "demo-memory-101"}
```

按 Ctrl+C 停止服务，再用同一条命令启动。第二轮：

```json
{"message": "请用计算工具把上一次结果加10。", "thread_id": "demo-memory-101"}
```

预期：结果 1091；`calculate` 参数为 `a=1081`、`b=10`；两轮后 `message_count=4`。

说明：重启会清空 `InMemorySaver`。没有内存状态时，程序从 SQLite 恢复已完成的用户消息与最终回答，再追加新问题。恢复的是对话历史，不是执行到一半的图节点，也不会重新执行历史工具。

## 用例三：MCP 工具业务错误

目的：验证工具错误被记录并解释，而不是让接口崩溃。

```json
{"message": "请实际调用calculate，参数a=10、b=0、operation=divide，用于验证错误处理。失败后解释错误，不要再次调用。", "thread_id": "demo-error-101"}
```

预期：trace 中 `source=mcp`、`status=error`、`attempts=1`；`plan` 中工具步骤为 `failed`，最终回答步骤为 `completed`；整体 `status=completed_with_errors`。

说明：除零是确定性业务错误，重试无意义。执行器只对 `ConnectionError` 做有限重试；超时后不会重复执行可能仍在运行的任务。如果模型没有调用工具，这次演示不算通过。
