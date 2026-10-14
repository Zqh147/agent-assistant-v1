# 项目二增量更新与本机验收

只更新新增/修改文件，不含 API Key、虚拟环境、SQLite 数据库或整套仓库。

## 1. 检查并应用

解压到 E:\google\agent-support-update。在空闲 PowerShell：

```powershell
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" "E:\google\agent-support-update\apply_agent_support_update.py" --target "E:\github\agent-assistant-v1"
```

检查通过后，停止项目二 8000 服务；项目一 8001 可保持运行。再应用：

```powershell
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" "E:\google\agent-support-update\apply_agent_support_update.py" --target "E:\github\agent-assistant-v1" --apply
```

完整预检后才复制；UTF-8 只忽略 LF/CRLF 差异，真实差异会停止。原修改文件在补丁 backup_* 中备份，代码应用保留 .env 和数据库。

## 2. 测试、持久配置

```powershell
Set-Location "E:\github\agent-assistant-v1"
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" -m pytest -q
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" -m app.configure_support --apply
```

预期 61 passed；新增业务无新增依赖。配置命令保留模型名、Key、工具传输和其他已有值；私有备份不进入 Git。配置后新开终端，以避免旧环境变量覆盖。

## 3. 离线执行验证

```powershell
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" -m app.support_evaluation --mode offline
```

预期 9/9。它验证执行，不是在线模型成绩。增加 --live-rag 使用真实项目一服务。

## 4. 启动与在线验收

新终端启动（保持项目一 8001 运行）：

```powershell
Set-Location "E:\github\agent-assistant-v1"
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

另一个空闲终端检查 /health，support_mode=demo、rag_backend=project1、online=True 后：

```powershell
Set-Location "E:\github\agent-assistant-v1"
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" -m app.check_support_service --require-online
& "E:\github\LearningHub\projects\Python\day21\.venv\Scripts\python.exe" -m app.support_evaluation --mode online
```

服务检查预期 10/10，在线业务评测预期 9/9，必须以本机实际输出为准。上传报告后再核对真实模型回答和工具决策。

## 5. 进程重启

按 SUPPORT_GUIDE 第一轮准备草稿，保存 thread_id 和工单号；停止并重启 8000 服务，再用原会话“确认提交”、查询进度。预期同一工单变 submitted，重复确认不生成新工单。助手端已真实停止/重启服务验证；Windows 本机仍待执行。
