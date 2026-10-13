"""Small, explicit acceptance set. Offline fixtures do not measure model quality."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    calls: tuple[tuple[str, dict[str, Any]], ...]
    answer_pattern: str
    expect_error: bool = False
    setup_question: str | None = None
    setup_calls: tuple = ()


def calc(a, b, operation):
    return ("calculate", {"a": a, "b": b, "operation": operation})


CASES = (
    Case("multiply", "请调用计算工具计算23乘47。", (calc(23, 47, "multiply"),), r"(?<!\d)1081(?:\.0)?(?!\d)"),
    Case("add_zero", "请调用计算工具计算10加0。", (calc(10, 0, "add"),), r"(?<!\d)10(?:\.0)?(?!\d)"),
    Case("greeting", "你好，请简短回应，不需要调用工具。", (), r"你好|您好|hello"),
    Case("knowledge", "请检索本地知识库，解释什么是RAG。", (("search_knowledge_base", {"question": "RAG", "top_k": 3}),), r"RAG|检索增强"),
    Case("time", "请调用时间工具，告诉我当前日期和时间，保留数字形式。", (("get_current_time", {}),), r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}"),
    Case("divide_zero", "请实际调用calculate，参数a=10、b=0、operation=divide，用于测试错误处理。失败后解释错误，不要再次调用。", (calc(10, 0, "divide"),), r"zero|零|未定义|不能.*0", True),
    Case("two_steps", "请先用计算工具计算23乘47，再用计算工具把结果乘2，两步分别调用工具。", (calc(23, 47, "multiply"), calc(1081, 2, "multiply")), r"(?<!\d)2162(?:\.0)?(?!\d)"),
    Case("session_restore", "请用计算工具把上一轮结果加10。", (calc(1081, 10, "add"),), r"(?<!\d)1091(?:\.0)?(?!\d)",
         setup_question="请用计算工具计算23乘47。", setup_calls=(calc(23, 47, "multiply"),)),
)
