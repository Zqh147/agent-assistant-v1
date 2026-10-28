"""HITL demo: real SupportService + temporary SQLite; no model calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

from langgraph.types import Command
from app.config import settings
from app.support_store import SupportStore
from app.support_service import SupportService
from app.multi_agent.ticket_approval_graph import build_ticket_approval_graph

assert settings.support_mode == 'demo'

with TemporaryDirectory(prefix='ticket-hitl-') as tmp:
    store = SupportStore(Path(tmp) / 'support.db', settings.demo_orders_path)
    service = SupportService(store, settings.support_rules_path)
    graph = build_ticket_approval_graph(service)
    customer_id = 'demo-customer-001'

    def prepare(thread):
        response = service.prepare_ticket(
            order_id='DEMO-1001', service_type='repair', issue_summary='设备频繁掉线',
            customer_id=customer_id, thread_id=thread,
            user_messages=['设备频繁掉线'],
        )
        assert response['status'] == 'draft_ready', response
        return response['ticket']['ticket_id']

    thread = 'business-' + uuid4().hex[:8]
    ticket_id = prepare(thread)
    config = {'configurable': {'thread_id': 'approval-' + uuid4().hex}}
    initial = {'ticket_id': ticket_id, 'customer_id': customer_id, 'thread_id': thread}
    paused = graph.invoke(initial, config=config)
    assert paused.get('__interrupt__'), paused
    assert store.get_ticket(ticket_id, customer_id)['status'] == 'draft'
    print('INTERRUPTED:', paused['__interrupt__'][0].value)
    print('Before approval: draft (PASS)')

    done = graph.invoke(Command(resume={'approved': True, 'ticket_id': ticket_id}), config=config)
    assert done['workflow_status'] == 'submitted', done
    assert store.get_ticket(ticket_id, customer_id)['status'] == 'submitted'
    names = [e['event'] for e in store.events(ticket_id, customer_id)]
    assert names.count('submitted') == 1
    print('Approved and submitted: PASS')
    print('Submitted events:', names.count('submitted'))

    reject_thread = 'business-' + uuid4().hex[:8]
    reject_id = prepare(reject_thread)
    reject_config = {'configurable': {'thread_id': 'approval-' + uuid4().hex}}
    reject_state = {'ticket_id': reject_id, 'customer_id': customer_id, 'thread_id': reject_thread}
    paused = graph.invoke(reject_state, config=reject_config)
    assert paused.get('__interrupt__')
    rejected = graph.invoke(
        Command(resume={'approved': False, 'ticket_id': reject_id}), config=reject_config,
    )
    assert rejected['workflow_status'] == 'rejected'
    assert store.get_ticket(reject_id, customer_id)['status'] == 'draft'
    print('Rejected without submission: PASS')

print('HITL checkpoint/interrupt/resume: PASS')
