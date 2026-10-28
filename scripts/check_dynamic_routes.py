
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

service = SupportService(
    store,
    settings.support_rules_path,
)

order_agent = OrderAgent(
    llm=llm,
    support_tools=SupportTools(service),
)

knowledge_agent = KnowledgeAgent(llm=llm)

supervisor = DynamicSupervisor(llm=llm)

public_scope = {
    "product_model": "Archer AX10",
    "hardware_version": "V1",
    "region": "US",
}

cases = [
    (
        "order_only",
        "请查询订单 DEMO-1001 的设备型号、硬件版本和地区。",
    ),
    (
        "knowledge_only",
        "Archer AX10 V1 US 在 AP 模式下是否支持 QoS？",
    ),
    (
        "order_knowledge",
        "请查询订单 DEMO-1001 的实际设备，"
        "并根据该设备的技术资料解释 AP 模式下的 QoS 限制。",
    ),
]

passed = 0
failures = []

for task_type, question in cases:
    print(f"\n{'=' * 55}")
    print("任务类型:", task_type)
    print("用户问题:", question)

    # 每个任务建立独立的执行状态
    tools = SupervisorTools(
        order_agent=order_agent,
        knowledge_agent=knowledge_agent,
        trusted_context={
            "customer_id": "demo-customer-001",
        },
        public_product_scope=(
            public_scope
            if task_type == "knowledge_only"
            else None
        ),
    )

    try:
        result = supervisor.run(
            user_question=question,
            tools=tools,
            task_type=task_type,
            max_steps=6,
        )

        trace = result.data.get("tool_trace", [])
        names = [item["tool"] for item in trace]

        print("执行状态:", result.status)
        print("错误:", result.error)
        print("Tool Trace:", trace)
        print("证据数量:", len(result.evidence))
        print("最终回答:", result.summary)

        valid = result.status == "success"

        if task_type == "order_only":
            valid = (
                valid
                and bool(names)
                and all(n == "call_order_agent" for n in names)
                and result.data.get("order_id") == "DEMO-1001"
            )

        elif task_type == "knowledge_only":
            valid = (
                valid
                and bool(names)
                and all(n == "call_knowledge_agent" for n in names)
                and result.data.get("scope_source") == "public_input"
                and bool(result.evidence)
            )

        elif task_type == "order_knowledge":
            valid = (
                valid
                and len(names) >= 2
                and names[0] == "call_order_agent"
                and "call_knowledge_agent" in names[1:]
                and result.data.get("order_id") == "DEMO-1001"
                and bool(result.evidence)
            )

        if valid:
            passed += 1
            print("验收结果: PASS")
        else:
            failures.append(task_type)
            print("验收结果: FAIL")

    except Exception as exc:
        failures.append(task_type)
        print("执行异常:", type(exc).__name__, str(exc))

print(f"\n{'=' * 55}")
print(f"真实联调通过: {passed}/{len(cases)}")
print("失败任务:", failures)

if failures:
    raise SystemExit(1)
