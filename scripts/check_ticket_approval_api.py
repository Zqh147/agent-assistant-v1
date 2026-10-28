"""Standalone FastAPI TestClient smoke for demo approval; no live LLM or RAG."""
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.support_store import SupportStore
from app.support_service import SupportService
from app.multi_agent.ticket_approval_api import ActorContext, create_ticket_approval_router

assert settings.support_mode == 'demo', 'Demo-only smoke test'

with TemporaryDirectory(prefix='approval-api-') as tmp:
    store = SupportStore(Path(tmp) / 'support.db', settings.demo_orders_path)
    service = SupportService(store, settings.support_rules_path)
    actor = ActorContext(customer_id='demo-customer-001', thread_id='test-session-'+uuid4().hex[:8])

    # THIS IS TEST AUTH ONLY. Real integration must use existing verified login/session.
    current_actor = {'value': actor}
    def get_test_actor():
        return current_actor['value']

    app = FastAPI()
    app.include_router(create_ticket_approval_router(service, actor_dependency=get_test_actor))
    client = TestClient(app)

    def draft(thread_id):
        result = service.prepare_ticket(
            'DEMO-1001', 'repair', '设备频繁掉线',
            customer_id=actor.customer_id, thread_id=thread_id,
            user_messages=['设备频繁掉线'],
        )
        assert result['status'] == 'draft_ready', result
        return result['ticket']['ticket_id']

    first_id = draft(actor.thread_id)
    start = client.post('/ticket-approvals', json={'ticket_id': first_id})
    assert start.status_code == 201, start.text
    approval_id = start.json()['approval_id']
    assert start.json()['status'] == 'pending'
    assert start.json()['approval']['ticket_id'] == first_id
    assert store.get_ticket(first_id, actor.customer_id)['status'] == 'draft'
    print('API approval created and interrupted: PASS')

    # Attempt to access another user's workflow must be rejected.
    current_actor['value'] = ActorContext('another-customer', actor.thread_id)
    assert client.get(f'/ticket-approvals/{approval_id}').status_code == 404
    assert client.post(f'/ticket-approvals/{approval_id}/decision', json={'approved': True}).status_code == 404
    current_actor['value'] = actor
    print('API cross-user access denied: PASS')

    pending = client.get(f'/ticket-approvals/{approval_id}')
    assert pending.status_code == 200 and pending.json()['status'] == 'pending'
    approve = client.post(f'/ticket-approvals/{approval_id}/decision', json={'approved': True})
    assert approve.status_code == 200, approve.text
    assert approve.json()['status'] == 'submitted'
    assert store.get_ticket(first_id, actor.customer_id)['status'] == 'submitted'
    assert [x['event'] for x in store.events(first_id, actor.customer_id)].count('submitted') == 1
    assert client.post(f'/ticket-approvals/{approval_id}/decision', json={'approved': True}).status_code == 409
    print('API resume approved, exactly one submission: PASS')

    second_thread = actor.thread_id + '-reject'
    current_actor['value'] = ActorContext(actor.customer_id, second_thread)
    second_id = draft(second_thread)
    start2 = client.post('/ticket-approvals', json={'ticket_id': second_id})
    assert start2.status_code == 201, start2.text
    reject_id = start2.json()['approval_id']
    reject = client.post(f'/ticket-approvals/{reject_id}/decision', json={'approved': False})
    assert reject.status_code == 200 and reject.json()['status'] == 'rejected'
    assert store.get_ticket(second_id, actor.customer_id)['status'] == 'draft'
    print('API decline preserves draft: PASS')

print('FastAPI HITL cross-request smoke: PASS')
