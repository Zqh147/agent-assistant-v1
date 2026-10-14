"""Persist only feature settings; preserve model credentials and other values."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil

FEATURE_VALUES = {
    "RAG_BACKEND": "project1", "RAG_SERVICE_URL": "http://127.0.0.1:8001",
    "SUPPORT_MODE": "demo", "SUPPORT_DB": "data/support_demo.db",
    "DEMO_CUSTOMER_ID": "demo-customer-001", "MAX_STEPS": "10",
    "TOOL_TIMEOUT_SECONDS": "45", "RAG_TIMEOUT_SECONDS": "30",
}


def configure(root, apply=False):
    root = Path(root)
    target = root / ".env"
    original = target.read_bytes() if target.exists() else b""
    text = original.decode("utf-8-sig")
    lines, seen = [], set()
    for line in text.splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        key = match.group(1) if match else None
        if key in FEATURE_VALUES:
            if key not in seen:
                lines.append(f"{key}={FEATURE_VALUES[key]}")
                seen.add(key)
        else:
            lines.append(line)
    for key, value in FEATURE_VALUES.items():
        if key not in seen:
            lines.append(f"{key}={value}")
    updated = ("\n".join(lines) + "\n").encode("utf-8")
    if not apply:
        return {"applied": False, "changed_keys": list(FEATURE_VALUES), "message": "仅检查；--apply 会更新业务接入参数，保留 API Key、模型名和其他已有配置。"}
    if updated == original:
        return {"applied": True, "changed": False, "message": "接入配置已符合要求。"}
    backup = None
    if target.exists():
        backup = root / "data/private_backups" / ("env_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    temporary = target.with_name(".env.support-tmp")
    temporary.write_bytes(updated)
    temporary.replace(target)
    return {"applied": True, "changed": True, "changed_keys": list(FEATURE_VALUES), "backup_created": backup is not None,
            "message": "接入配置已保存；请新开终端并重启项目二，使新配置生效。"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    import json
    root = Path(__file__).resolve().parent.parent
    print(json.dumps(configure(root, args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
