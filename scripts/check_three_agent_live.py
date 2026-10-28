"""Real-LLM multi-agent smoke test. Writes ONLY to an isolated temporary SQLite DB."""
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

from app.config import settings
from app.llm_client import LLMClient
from app.support_store import SupportStore
from app.support_service import SupportService
from app.support_tools import SupportTools
from app.multi_agent.order_agent import OrderAgent
from app.multi_agent.knowledge_agent import KnowledgeAgent
from app.multi_agent.ticket_agent import TicketAgent
from app.multi_agent.supervisor_tools import SupervisorTools
from app.multi_agent.dynamic_supervisor import DynamicSupervisor

assert settings.support_mode == 'demo', 'Only demo business mode is supported'
assert settings.rag_backend == 'project1', 'Start Project 1 RAG service first'

llm = LLMClient()
order_agent = None
knowledge_agent = KnowledgeAgent(llm=llm)
supervisor = DynamicSupervisor(llm=llm)

cases = [
    (
        'order_ticket',
        '我有订单 DEMO-1001，设备频繁掉线。请为该订单准备维修工单草稿。'
        '申请类型为 repair，故障描述请逐字摘录“设备频繁掉线”。请勿正式提交。',
        ['call_order_agent', 'call_ticket_agent'],
    ),
    (
        'order_knowledge_ticket',
        '订单 DEMO-1001 的设备频繁掉线。请先查询这个订单对应设备的掉线排查资料，'
        '再为该订单准备维修工单草稿。申请类型为 repair，故障描述请逐字摘录“设备频繁掉线”。'
        '请勿正式提交。',
        ['call_order_agent', 'call_knowledge_agent', 'call_ticket_agent'],
    ),
]

failures = []
for i, (task_type, question, expected_calls) in enumerate(cases, start=1):
    with TemporaryDirectory(prefix='three-agent-smoke-') as tmp:
        store = SupportStore(Path(tmp) / 'support.db', settings.demo_orders_path)
        service = SupportService(store, settings.support_rules_path)
        order_agent = OrderAgent(llm=llm, support_tools=SupportTools(service))
        ticket_agent = TicketAgent(llm=llm, support_tools=SupportTools(service))

        thread_id = f'live-three-agent-{i}'
        tools = SupervisorTools(
            order_agent=order_agent,
            knowledge_agent=knowledge_agent,
            ticket_agent=ticket_agent,
            trusted_context={
                'customer_id': 'demo-customer-001',
                'thread_id': thread_id,
                'user_messages': [question],
            },
        )
        try:
            result = supervisor.run(
                user_question=question,
                tools=tools,
                task_type=task_type,
                max_steps=8,
            )
            trace = result.data.get('tool_trace', [])
            names = [item['tool'] for item in trace]
            ticket_id = result.data.get('ticket_id')
            saved = store.get_ticket(ticket_id, 'demo-customer-001') if ticket_id else None
            events = store.events(ticket_id, 'demo-customer-001') if ticket_id else []
            event_names = [item['event'] for item in events]
            checks = {
                "supervisor_pending_confirmation": (
                    result.status == "pending_confirmation"
                ),
                "agent_execution_order": (
                    names == expected_calls
                ),
                "ticket_written_to_sqlite": (
                    saved is not None
                ),
                "ticket_remains_draft": (
                    saved is not None and saved["status"] == "draft"
                ),
                "matches_order": (
                    saved is not None and saved["order_id"] == "DEMO-1001"
                ),
                "matches_thread": (
                    saved is not None and saved["thread_id"] == thread_id
                ),
                "correct_service_type": (
                    saved is not None and saved["service_type"] == "repair"
                ),
                "correct_issue_summary": (
                    saved is not None
                    and saved["issue_summary"] == "设备频繁掉线"
                ),
                "draft_created_once": (
                    event_names.count("draft_created") == 1
                ),
                "not_submitted": (
                    "submitted" not in event_names
                ),
            }
            if task_type == 'order_knowledge_ticket':
                checks["rag_has_evidence"] = bool(result.evidence)
            valid = all(checks.values())
            print(f'\nCASE: {task_type}')
            print('STATUS:', result.status)
            print('TOOL_TRACE:', trace)
            print('TICKET_ID:', ticket_id)
            print('DATABASE_STATUS:', saved['status'] if saved else None)
            print('EVENTS:', event_names)
            print('EVIDENCE_COUNT:', len(result.evidence))
            print(
                "FAILED_CHECKS:",
                [name for name, passed in checks.items() if not passed]
            )
            print('RESULT:', 'PASS' if valid else 'FAIL')
            if not valid:
                print('SUMMARY:', result.summary)
                print('ERROR:', result.error)
                failures.append(task_type)
        except Exception as exc:
            print(f'CASE: {task_type}; ERROR: {type(exc).__name__}: {exc}')
            failures.append(task_type)

print(f'\nReal three-agent smoke: {len(cases) - len(failures)}/{len(cases)}')
if failures:
    raise SystemExit(f'Failed cases: {failures}')
