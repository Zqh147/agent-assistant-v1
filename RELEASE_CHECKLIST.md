# 发布前检查

目的：保证说明与实现一致，让其他人能安装和复现演示。
本步骤不创建远程仓库，也不执行 git push。

## 在 Windows 项目根目录检查

```powershell
python --version
python -c "import sys; print(sys.executable)"
python -m pip check
python -m pytest -q
python -m app.evaluation --mode offline --transport mcp
git status --short
```

如果 git status 显示上级 LearningHub 的大量文件，说明当前使用的是学习仓库。发布公开项目时应先建立独立目录/仓库，确认 git rev-parse --show-toplevel，避免推送整个学习记录。

## 文件范围

应该包含：app、tests、data/knowledge.md、requirements.txt、README、说明文档、.env.example、.gitignore、.dockerignore、Dockerfile。
不应提交：.env、虚拟环境、会话数据库及 journal/wal/shm 文件、缓存、临时报告、更新压缩包。
.gitignore 只影响尚未追踪的文件；已追踪文件不会自动从历史移除。

```powershell
git check-ignore .env .venv/test data/sessions.db data/sessions.db-wal reports/test.json patch.zip
git ls-files .env 'data/*.db' 'data/*.db-*'
git diff --stat
```

第一条应列出忽略项，第二条理想情况下没有输出。检查暂存差异后再提交；不要直接对不明确的上级仓库执行 git add .。
已经泄露的密钥需要在服务商处撤销，仅删除文件不能撤销泄露。

## 验收与可复现性

25项测试、8题离线和在线验收是目前的已报告结果，不是性能保证。
新电脑先跑离线，确认环境后再主动运行在线评估。保留真实模型名、运行日期及依赖版本供自己复盘。
依赖范围尚未锁死，Docker尚未在本次环境实测；发布时保持说明准确。
演示环境使用单进程；尚未验证并发同会话、多进程状态一致性。
真实报告发布前人工检查内容；默认 reports 被忽略，可另整理不含个人数据的示例报告。
