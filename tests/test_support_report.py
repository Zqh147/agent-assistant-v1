import json
import httpx
from app.check_support_service import check


def test_failed_live_check_keeps_partial_turns_and_thread_without_provider_body(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        if request.url.path == '/health':
            return httpx.Response(200, json={'status':'ok','support_mode':'demo','rag_backend':'project1','online':True})
        if request.url.path == '/tools':
            return httpx.Response(200, json=['lookup_order','search_after_sales_rules','assess_after_sales','prepare_ticket','submit_ticket','get_ticket','search_knowledge_base'])
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, headers={'content-type':'application/json; charset=utf-8'}, json={'answer':'请提供订单号','tool_trace':[],'errors':[],'status':'completed'})
        return httpx.Response(503, json={'detail':'sensitive-provider-body'})
    original = httpx.Client
    monkeypatch.setattr(httpx,'Client', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    output=tmp_path/'failed.json'
    report=check(url='http://local-fixture',require_online=True,output=output)
    assert report['exception_type'] == 'HTTPStatusError'
    assert report['thread_id'].startswith('support-http-')
    assert len(report['turns']) == 2
    assert report['turns'][-1]['http_status'] == 503
    assert report['summary']['passed'] < report['summary']['checks']
    assert not report['checks']['all_turns_completed']
    assert 'sensitive-provider-body' not in output.read_text(encoding='utf-8')
    assert json.loads(output.read_text(encoding='utf-8'))['exception_type'] == 'HTTPStatusError'
