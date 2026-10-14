# 网络设备售后咨询与工单处理 Agent

新增业务场景已实现订单查询、演示规则资格判断、工单草稿、确认提交和进度查询，并接入项目一的产品检索服务。开始运行前请阅读 [售后场景说明](SUPPORT_GUIDE.md) 与 [增量更新指南](SUPPORT_UPDATE_GUIDE.md)。

示例订单、售后规则和工单均为本地演示数据，不代表厂商正式政策或真实退款/维修办理。下文保留原有通用 Agent 能力说明；新增业务以 SUPPORT_GUIDE.md 为准。

基于 Python、LangGraph、DeepSeek 和 MCP 的个人学习项目。支持模型选择工具、连续工具调用、业务会话恢复，以及可重复运行的验收评估。

## 已实现能力与边界

| 能力 | 实现 | 当前边界 |
| --- | --- | --- |
| Agent 循环 | LLM 决策、工具执行、结果回传与条件路由 | 设置模型调用步数上限；不是无限自主执行 |
| MCP | stdio 子进程、tools/list 动态注册、tools/call 调用 | 当前接入计算和时间工具，每次操作创建会话，未做连接池 |
| 执行步骤 | 根据真实 tool_calls 更新 plan，与 trace 用 call_id 对应 | 属于增量执行记录，不是独立的预先任务拆解器 |
| 会话 | InMemorySaver 保存进程内状态；SQLite 保存完成的对话轮次及业务上下文 | 支持聊天历史与待确认工单恢复；完整图节点检查点仍在内存中 |
| 知识检索 | 本地适配器通过 HTTP 调用项目一的混合检索与重排服务，保留证据与设备范围 | RAG_BACKEND=demo 时保留旧关键词示例；Embedding/向量库等在项目一执行 |
| 错误处理 | 业务错误一次失败，ConnectionError 最多两次，工具超时停止等待 | 不能强制终止运行中的 Python 线程；其他传输异常不保证重试 |
| 评估 | 独立离线/在线模式、临时数据库、逐题 JSON 报告 | 八题验收集，不代表业务准确率；答案采用关键词/正则检查 |

## Windows PowerShell 从零启动

在包含 requirements.txt 的项目根目录操作。已有可用虚拟环境时无需重建。
开发验证覆盖 Windows Python 3.11（用户本机）和 Linux Python 3.12。

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

若 PowerShell 不允许激活脚本，可直接使用虚拟环境解释器：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

下面命令默认已激活环境。VS Code 的 Python: Select Interpreter 选择本项目 .venv 中的 python.exe。
首次创建配置；已有 .env 时直接编辑，不重复复制覆盖：

```powershell
Copy-Item .env.example .env
```

配置示例：

```dotenv
DEEPSEEK_API_KEY=填入自己的Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
TOOL_TRANSPORT=mcp
MAX_STEPS=6
TOOL_TIMEOUT_SECONDS=15
SESSION_DB=data/agent_sessions.db
```

模型名按账号可用配置填写。没有 API Key 时，普通聊天入口使用有限的离线规则；在线评估则会拒绝运行。
现有 KNOWLEDGE_PATH 配置尚未接入检索函数，当前请直接编辑 data/knowledge.md。

```powershell
python -m pytest -q
python -m uvicorn app.api:app --reload
```

浏览器打开 http://127.0.0.1:8000/docs 。终端保持运行属于正常现象，Ctrl+C 停止服务。
修改 .env 后手动重启服务。MCP 子进程由程序自动启动，不必单独打开另一个终端启动 server。

macOS/Linux 可使用 python3 -m venv .venv，再执行 source .venv/bin/activate；其余 Python 命令相同。

## 接口与命令行

| 接口 | 用途 |
| --- | --- |
| GET /health | 进程健康检查，不验证模型凭据或 MCP 连通性 |
| GET /tools | 已注册工具名称 |
| GET /threads | SQLite 已保存会话列表 |
| POST /chat | message 与 thread_id 输入，返回 answer、trace、plan 等 |

POST /chat 示例：

```json
{"message":"请用计算工具计算23乘47。","thread_id":"demo-new-001"}
```

检查 answer 包含 1081，tool_trace 包含 calculate，MCP 模式下 source 为 mcp。
CLI 入口为 python -m app.cli，输入 exit 退出。
当前没有登录鉴权和多租户权限隔离；thread_id 是会话标识，不是权限凭证。按本地单进程演示使用，不直接公开聊天历史接口。

## 评估

无需启动 API，分别执行：

```powershell
python -m app.evaluation
python -m app.evaluation --mode offline --transport mcp
python -m app.evaluation --mode online --transport mcp --case multiply
python -m app.evaluation --mode online --transport mcp
```

默认 offline/local，与 .env 的 TOOL_TRANSPORT 无关。offline 使用脚本模型验证执行框架；online 才检验真实模型行为并消耗 API 用量。
每次使用临时数据库和新会话 ID，报告写入 reports/evaluation-模式-传输-运行ID.json。
退出码 0=全部通过，1=存在失败，2=参数或配置错误。

2026-10-13 开发验收：Windows 本机用户反馈 25 个自动化测试通过；本地离线、MCP 离线、MCP 在线三组八题均为 8/8。这是有限验收集的一次运行结果，模型版本与依赖变化可能改变结果；请自行重新运行。
依赖使用 requirements.txt 中的版本范围，尚未提供跨平台精确锁文件。
详见 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)。

## Docker（可选）

Docker 配置已整理，尚未在本次环境构建运行验证。安装 Docker 后可以执行：

```powershell
docker build -t agent-assistant .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env -e SESSION_DB=/state/sessions.db -v agent-state:/state agent-assistant
```

.env 只在运行时注入；会话数据库存入命名卷，停止容器后仍保留。宿主机数据库和 API Key 不放进镜像。

## 代码阅读顺序

1. app/api.py：请求入口与字段校验。
2. app/graph.py：共享 State、节点与循环路由。
3. app/llm_client.py：模型调用及 LangChain/API 消息格式转换。
4. app/tools.py 与 app/mcp_client.py：本地执行和 MCP 执行。
5. app/planner.py：根据实际调用维护步骤状态。
6. app/storage.py：事务与连接释放。
7. app/evaluation.py：验收标准、执行与报告。

演示和面试讲解见 [DEMO_GUIDE.md](DEMO_GUIDE.md)；发布前检查见 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)。
