
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from app.config import settings
from app.llm_client import LLMClient
from app.support_store import SupportStore
from app.support_service import SupportService
from app.support_tools import SupportTools
from app.multi_agent.order_agent import OrderAgent
from app.multi_agent.knowledge_agent import KnowledgeAgent
from app.multi_agent.supervisor_tools import SupervisorTools
from app.multi_agent.dynamic_supervisor import DynamicSupervisor

assert settings.support_mode == "demo"
assert settings.rag_backend == "project1"

llm = LLMClient()

store = SupportStore(
    settings.support_db,
    settings.demo_orders_path,
)
service = SupportService(store, settings.support_rules_path)
support_tools = SupportTools(service)

order_agent = OrderAgent(llm=llm, support_tools=support_tools)
knowledge_agent = KnowledgeAgent(llm=llm)

tools = SupervisorTools(
    order_agent=order_agent,
    knowledge_agent=knowledge_agent,
    trusted_context={"customer_id": "demo-customer-001"},
)

supervisor = DynamicSupervisor(llm=llm)

question = (
    "请先查询订单 DEMO-1001 的设备型号、硬件版本和地区，"
    "再根据该订单实际设备的技术资料，"
    "解释 AP 模式下为什么不能使用 QoS。"
)

print("开始真实双 Agent 联调...")

result = supervisor.run(
    user_question=question,
    tools=tools,
    max_steps=6,
)

print("\n执行状态:", result.status)
print("最终回答:", result.summary)
print("错误信息:", result.error)
print("已验证订单:", tools.verified_order_id)
print("设备范围:", tools.verified_product_scope)

print("\nTool Trace:")
for item in result.data.get("tool_trace", []):
    print(item)

print("\n证据数量:", len(result.evidence))

if result.status == "success":
    assert result.data["order_id"] == "DEMO-1001"
    assert result.evidence
    assert [x["tool"] for x in result.data["tool_trace"]][:2] == [
        "call_order_agent",
        "call_knowledge_agent",
    ]
    print("\n真实多 Agent 联调通过")
else:
    print("\n真实联调未通过，需要进一步排查")
