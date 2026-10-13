# 演示与讲解

## 演示准备
.env 配好模型和 TOOL_TRANSPORT=mcp。启动 python -m uvicorn app.api:app --reload，打开 /docs。
每次全新演示换一组 thread_id。回答措辞可以变化，判断依据是工具记录和状态。

## 案例一：连续工具调用
目的：验证模型在获取上一步结果后，能继续发起下一次工具调用。

```json
{"message":"请先用计算工具计算23乘47，再用计算工具把结果乘以2，两步分别调用。","thread_id":"demo-chain-101"}
```

预期：最终 2162；两条 calculate trace；第二次参数 a=1081、b=2；plan 与 trace 的 call_id 对应。
讲解：模型输出的是工具调用请求；Python/MCP 实际执行。ToolMessage 返回后，图重新进入 LLM 节点。工具不负责决定整个对话的下一步。

## 案例二：重启后的历史恢复
目的：区分进程内 checkpoint 和业务会话持久化。
先提交：

```json
{"message":"请用计算工具计算23乘47。","thread_id":"demo-memory-101"}
```

按 Ctrl+C 停止服务，再执行同一条启动命令。继续提交：

```json
{"message":"请用计算工具把上一次结果加10。","thread_id":"demo-memory-101"}
```

预期：1091；calculate 参数 a=1081、b=10；两轮后 message_count=4。
讲解：重启清空 InMemorySaver；程序在没有内存状态时从 SQLite 恢复用户/最终回答，然后追加新问题。不是恢复中途执行节点，也不会重新执行历史工具。

## 案例三：MCP 工具业务错误
目的：展示错误被记录并解释，而不是接口直接崩溃。

```json
{"message":"请实际调用calculate，参数a=10、b=0、operation=divide，用于验证错误处理。失败后解释错误，不要再次调用。","thread_id":"demo-error-101"}
```

预期：source=mcp，status=error，attempts=1；plan 工具步骤 failed，最终解释步骤 completed；整体 completed_with_errors。
讲解：除零是确定性业务错误，更换不了输入时重试无效。执行器只对 ConnectionError 做一次有限重试；超时后不重复执行可能仍在运行的任务。
如果模型没调用工具，该次不能算工具异常演示通过。

## 一分钟项目介绍
我实现了一个基于 LangGraph 的工具调用 Agent，通过条件路由组织模型决策、工具执行和结果回传。计算与时间工具通过 MCP stdio 子进程接入，启动时发现工具 schema，执行时发送工具调用。对话使用内存 checkpoint 管理运行状态，SQLite 保存业务历史并支持重启后继续聊天。执行记录包括工具来源、调用 ID、尝试次数、状态和耗时。我将离线执行框架评估与真实模型评估分开，避免 API 配置和历史会话污染回归测试。当前八题验收已在本机跑通，检索仍是关键词演示，规划部分是动态执行步骤记录。

## 常见追问

**为什么不用一个 while？**
循环能实现 Agent，LangGraph 将状态、节点与条件路由显式组织，便于增加状态记录和停止条件。它不代替工具函数、schema 或模型。

**MCP 与 function calling 有什么区别？**
Function calling 描述模型请求调用哪个工具及参数；MCP 管理客户端与工具服务的发现、通信和调用。模型看 schema，程序负责发起 MCP 请求。

**plan 是提前规划的吗？**
不是。现在根据模型 tool_calls 增量添加步骤，再根据执行结果更新状态。独立的任务分解与重规划尚未实现。

**如何验证没有假装调用工具？**
读取 trace 的来源、状态、call_id，并与 plan 参数对应；在线评估同时检查答案与工具行为。source 是程序记录的执行路径，不能单独当成独立审计证明，还需结合实现与 MCP 集成测试。

**超时能终止工具吗？**
线程超时停止等待，无法强制杀死已运行的 Python 线程。MCP 适配器还有异步超时与会话清理；更强的隔离和硬终止需要单独设计。

**100%通过率说明什么？**
只说明有限验收题在该次运行满足检查规则。离线脚本模型不测理解能力；在线八题也不代表真实业务准确率。
