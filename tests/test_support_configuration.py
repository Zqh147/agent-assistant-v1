from pathlib import Path
from app.configure_support import configure


def test_configuration_preserves_credentials_model_and_unrelated_values(tmp_path):
    original = b'# configuration\r\nDEEPSEEK_API_KEY=local-test-secret\r\nDEEPSEEK_MODEL=chosen-model\r\nTOOL_TRANSPORT=mcp\r\nOTHER_SETTING=keep\r\nRAG_BACKEND=demo\r\nRAG_BACKEND=demo\r\n'
    target = tmp_path / '.env'
    target.write_bytes(original)
    dry = configure(tmp_path)
    assert not dry['applied']
    assert target.read_bytes() == original
    result = configure(tmp_path, apply=True)
    updated = target.read_text()
    assert result['backup_created']
    for line in ['DEEPSEEK_API_KEY=local-test-secret', 'DEEPSEEK_MODEL=chosen-model', 'TOOL_TRANSPORT=mcp', 'OTHER_SETTING=keep']:
        assert line in updated
    assert updated.count('RAG_BACKEND=') == 1
    assert 'RAG_BACKEND=project1' in updated
    assert next((tmp_path/'data/private_backups').iterdir()).read_bytes() == original
    assert not configure(tmp_path, apply=True)['changed']


def test_configuration_can_create_feature_only_file(tmp_path):
    configure(tmp_path, apply=True)
    text = (tmp_path/'.env').read_text()
    assert 'SUPPORT_MODE=demo' in text
    assert 'DEEPSEEK_API_KEY' not in text


def test_invalid_encoding_does_not_overwrite_existing_file(tmp_path):
    import pytest
    target=tmp_path/'.env'
    target.write_bytes(b'\xffinvalid')
    with pytest.raises(UnicodeDecodeError):
        configure(tmp_path, apply=True)
    assert target.read_bytes() == b'\xffinvalid'
