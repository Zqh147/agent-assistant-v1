
import json

from app.config import settings
from app.llm_client import LLMClient
from app.rag_tool import search_knowledge_base
from app.multi_agent.contracts import AgentTask
from app.multi_agent.knowledge_agent import KnowledgeAgent


# 1. 检查真实运行条件
llm = LLMClient()

if not llm.online:
    raise RuntimeError("请先配置真实 DeepSeek API Key")

if settings.rag_backend != "project1":
    raise RuntimeError("请设置 RAG_BACKEND=project1")


# 2. 包装原有检索工具，记录实际调用
retrieval_count = 0

def traced_retrieve(**kwargs):
    global retrieval_count
    retrieval_count += 1

    print(f"\n第 {retrieval_count} 次真实 RAG 调用")
    print("检索参数:", kwargs)

    result = search_knowledge_base(**kwargs)

    print("检索状态:", result.get("status"))
    print("证据数量:", len(result.get("results", [])))

    return result


# 3. 创建独立 Knowledge Agent
agent = KnowledgeAgent(
    llm=llm,
    retrieve=traced_retrieve,
)


# 4. 构造一个真实技术检索任务
task = AgentTask(
    task=(
        "请检索知识库并回答："
        "Archer AX10 V1 US 在 AP 模式下"
        "是否支持 NAT 转发？"
        "请根据证据回答，并说明资料来源。"
    ),
    product_scope={
        "product_model": "Archer AX10",
        "hardware_version": "V1",
        "region": "US",
    },
)


# 5. 执行 Agent Loop
result = agent.run(task, max_steps=4)


# 6. 查看执行结果
print("\n========== Agent 执行结果 ==========")
print("真实检索次数:", retrieval_count)
print("Agent 状态:", result.status)
print("最终回答:", result.summary)
print("证据数量:", len(result.evidence))
print("错误信息:", result.error)

print("\n========== 结构化结果 ==========")
print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
