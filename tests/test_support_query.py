import json
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from app.support_offline import route_demo


def test_business_prefix_is_not_forwarded_as_technical_query():
    question = '查询订单 DEMO-1001，然后检索 Wi-Fi 掉线的排查资料并注明来源。'
    context = {'order_id': 'DEMO-1001'}
    messages = [SystemMessage(content='服务端业务上下文：'+json.dumps(context)), HumanMessage(content=question),
                ToolMessage(content=json.dumps({'status':'found'}), name='lookup_order', tool_call_id='lookup')]
    response = route_demo(messages)
    assert response.tool_calls[0]['args']['question'] == 'Wi-Fi 掉线的排查资料'
    assert 'DEMO-' not in response.tool_calls[0]['args']['question']
