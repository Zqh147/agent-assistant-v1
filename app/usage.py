"""Per-request accounting of model calls and token usage.

A request handler opens ``track_usage()``; every ``LLMClient.invoke`` made
while it is open adds to the same meter, no matter which agent made the call.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from typing import Any, Iterator


@dataclass
class UsageMeter:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # False once any call returned no usage block (offline client, provider gap).
    tokens_complete: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage: Any) -> None:
        self.llm_calls += 1
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if not isinstance(prompt, int) or not isinstance(completion, int):
            self.tokens_complete = False
            return
        self.prompt_tokens += prompt
        self.completion_tokens += completion

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total_tokens": self.total_tokens}


_current: ContextVar[UsageMeter | None] = ContextVar("llm_usage_meter", default=None)


@contextmanager
def track_usage() -> Iterator[UsageMeter]:
    meter = UsageMeter()
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


def record_llm_call(usage: Any = None) -> None:
    """Called by LLMClient after each model decision; no-op outside track_usage."""
    meter = _current.get()
    if meter is not None:
        meter.add(usage)
