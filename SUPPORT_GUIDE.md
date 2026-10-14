# 网络设备售后咨询与工单处理 Agent

项目一负责找到适用设备的手册/FAQ 证据，项目二将产品知识、订单与演示售后规则组合成申请流程。订单、售后规则和工单均为本地演示数据；submitted 表示本地工单提交待人工审核，不表示退款、维修或向厂商发送申请。

## 1. 业务流程

缺订单号先追问 → lookup_order 查订单和设备 → 技术问题交给 search_knowledge_base → 申请问题交给 search_after_sales_rules 与 assess_after_sales → 资格通过且有问题描述时 prepare_ticket 保存草稿 → 展示详情并等待用户核对 → 本轮用户明确回复“确认提交”后 submit_ticket → get_ticket 查询实际记录。

纯技术/订单咨询不创建工单。已查订单的设备范围自动传给检索；显式范围冲突会停止。技术 query 不混入查询订单或创建工单等业务指令。

## 2. 组件分工

| 文件 | 作用 |
|---|---|
| app/graph.py | LangGraph 模型/工具循环，保存并使用订单、设备、待确认工单上下文 |
| app/rag_tool.py | 本地 Python 适配器通过 HTTP 请求项目一，只返回证据 |
| app/support_store.py | SQLite 示例订单、工单与事件；事务、唯一草稿键及提交幂等 |
| app/support_service.py | 根据记录与规则确定资格；提交前再次核对资格和版本 |
| app/support_tools.py | 6 个业务 Schema；客户、会话和确认权限由服务端注入 |
| app/storage.py | 兼容迁移原会话表，保存完成轮次和业务上下文 |
| app/llm_client.py | 在线 Function Calling；无 Key 时为明确标记的离线规则演示 |

已有 MCP 计算/时间工具保留动态发现和 stdio 调用。新增业务工具本地执行，RAG 为本地适配器 + HTTP；这些入口不能统一称为 MCP 接入。

## 3. 示例订单与规则

默认客户 demo-customer-001。初次建库时：DEMO-1001 为交付 3 天的 AX10 V1 US；DEMO-1002 为交付 90 天的 AX20 V1 US；DEMO-1003 为交付 400 天的 AX10；DEMO-1004 属于另一演示客户；DEMO-1005 已发货、未交付。日期按第一次启动时生成，之后重启不重置。

demo-support-v1 配置的示例维修窗口为交付后 365 天，退货为 7 天。这些数字是自建演示规则，不是 TP-Link、平台政策或法律结论。判断通过只允许提交申请，最终处理待人工审核。

同一客户/会话/订单/申请类型复用一张工单；重复提交不重复记事件。再次 prepare 返回原工单详情，不能将新的文字描述当成已修改成功。草稿编辑、取消、后台审核及真实退款/物流对接属于后续功能。

## 4. 安装与启动

沿用已有共享虚拟环境，无新增依赖。增量安装不包含 .env、虚拟环境或数据库。配置命令只更新业务参数，保留 Key、模型名、工具传输与其他配置，在本机 data/private_backups/ 保存原 .env；此目录已忽略于 Git。

```powershell
Set-Location "E:\github\agent-assistant-v1"
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" -m app.configure_support --apply
```

新开终端，保持项目一 8001 服务运行，再启动项目二：

```powershell
Set-Location "E:\github\agent-assistant-v1"
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

CLI：python -m app.cli；接口文档：http://127.0.0.1:8000/docs。GET /health 显示 online、support_mode、rag_backend、tool_transport；online 仅表示客户端配置，真实凭据需要实际调用验证。API 成功、业务错误和参数校验响应都声明 UTF-8，兼容旧 Windows PowerShell。

## 5. 多轮请求

POST /chat 输入 message、thread_id，可选演示 customer_id；返回 answer、tool_trace、plan、errors、business_context。GET /tools 查看工具，GET /demo/orders 查看默认客户示例订单，GET /threads 查看会话列表。

先新建会话并准备草稿：

```powershell
$threadId = "support-" + ([guid]::NewGuid()).ToString("N")
$body = @{
  message = "我的示例订单是 DEMO-1001，设备无法开机。请查询订单与演示维修规则，判断资格；符合时准备工单草稿，先展示给我核对，不要提交。"
  thread_id = $threadId
} | ConvertTo-Json
$result = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 180
$result | ConvertTo-Json -Depth 20
```

核对真实返回的订单、申请类型、问题描述和工单号。第二轮同一 thread_id，message 为“确认提交”；第三轮同一会话发送“查询工单进度”。服务端只接受当前轮的明确确认（可带实际工单号），模糊、引用和否定表达不授权提交。客户不能中途换绑同一会话。

## 6. 验收与记录

```powershell
python -m pytest -q
python -m app.support_evaluation --mode offline
python -m app.support_evaluation --mode offline --live-rag
python -m app.support_evaluation --mode online
python -m app.check_support_service --require-online
```

默认离线业务评测使用确定性规则与产品检索夹具，不代表 LLM 能力；--live-rag 使用真实项目一检索。在线必须配置 Key，不以离线模式替代。业务评测在隔离临时数据库运行，报告含完整回答、工具记录及工单快照；服务检查在本地演示服务中创建测试工单。报告写入 reports/，旧报告不覆盖。

workflow 自动检查只针对有限业务执行，不自动给回答语义准确率、覆盖率或幻觉率打分。原通用评测 python -m app.evaluation 保留，执行时关闭业务模块并使用原有用例。

## 7. 当前边界

本机单进程演示，没有真实客户登录鉴权；customer_id 是演示参数，不是生产身份凭证，归属检查不能等同于多租户认证。保存完成轮次与业务进度，完整 LangGraph 图检查点仍在内存；plan 来自实际工具调用增量记录，不是独立预先拆解规划器。Docker 文件保留，本次不声明新增场景已完成部署。项目一资料、门限与其评测成绩不因本次更新而改变。
