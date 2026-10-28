"""Human-approved ticket submission using LangGraph checkpoint + interrupt."""
from collections.abc import Mapping
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class TicketApprovalState(TypedDict, total=False):
    ticket_id: str
    customer_id: str
    thread_id: str
    ticket: dict[str, Any]
    workflow_status: str
    business_result: dict[str, Any]


def build_ticket_approval_graph(service, checkpointer=None):
    """Caller MUST supply authenticated customer/thread; demo uses in-memory checkpoints."""

    def validate_draft(state: TicketApprovalState) -> dict[str, Any]:
        ticket_id = state['ticket_id']
        customer_id = state['customer_id']
        thread_id = state['thread_id']
        found = service.get_ticket(ticket_id, customer_id=customer_id)
        ticket = found.get('ticket') if found.get('status') == 'found' else None
        if not isinstance(ticket, dict) or ticket.get('thread_id') != thread_id:
            return {'workflow_status': 'not_found', 'business_result': found}
        if ticket.get('status') == 'submitted':
            return {'workflow_status': 'already_submitted', 'ticket': ticket}
        if ticket.get('status') != 'draft':
            return {'workflow_status': 'invalid_ticket_status', 'ticket': ticket}
        return {'workflow_status': 'awaiting_approval', 'ticket': ticket}

    def approval(state: TicketApprovalState) -> dict[str, Any]:
        ticket = state['ticket']
        decision = interrupt({
            'action': 'submit_ticket',
            'ticket_id': state['ticket_id'],
            'order_id': ticket['order_id'],
            'service_type': ticket['service_type'],
            'issue_summary': ticket['issue_summary'],
            'prompt': '请核对草稿，明确选择批准或拒绝。',
        })
        if (
            not isinstance(decision, Mapping)
            or decision.get('approved') is not True
            or decision.get('ticket_id') != state['ticket_id']
        ):
            return {'workflow_status': 'rejected'}

        # Re-check identity/session AND eligibility after approval, before DB write.
        latest = service.get_ticket(state['ticket_id'], customer_id=state['customer_id'])
        actual = latest.get('ticket') if latest.get('status') == 'found' else None
        if not isinstance(actual, dict) or actual.get('thread_id') != state['thread_id']:
            return {'workflow_status': 'not_found', 'business_result': latest}
        result = service.submit_ticket(
            state['ticket_id'],
            customer_id=state['customer_id'],
            thread_id=state['thread_id'],
            confirmed_ticket_id=state['ticket_id'],
        )
        return {'workflow_status': result['status'], 'business_result': result}

    builder = StateGraph(TicketApprovalState)
    builder.add_node('validate_draft', validate_draft)
    builder.add_node('approval', approval)
    builder.add_edge(START, 'validate_draft')
    builder.add_conditional_edges(
        'validate_draft',
        lambda state: 'approval' if state['workflow_status'] == 'awaiting_approval' else END,
    )
    builder.add_edge('approval', END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
