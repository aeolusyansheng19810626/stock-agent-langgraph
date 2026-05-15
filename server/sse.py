"""SSE bridge: thread-safe event queue → async generator yielding SSE frames.

The graph runs in a worker thread (LangGraph stream is sync). Events are pushed
to a queue.Queue from the worker; the FastAPI endpoint pulls from the queue in
its async loop via run_in_executor and yields SSE-encoded strings.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable

# Sentinel pushed by the worker to signal end-of-stream.
_STREAM_END = object()


@dataclass
class EventEmitter:
    """Thread-safe event sink. Worker calls .emit(event, data); endpoint reads via .frames()."""

    q: queue.Queue = field(default_factory=queue.Queue)

    def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        self.q.put((event, data or {}))

    def close(self) -> None:
        self.q.put(_STREAM_END)

    async def frames(self) -> AsyncIterator[str]:
        """Async generator yielding SSE-formatted strings."""
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, self.q.get)
            if item is _STREAM_END:
                break
            event, data = item
            payload = json.dumps(data, ensure_ascii=False, default=str)
            # SSE frame: event: <name>\ndata: <json>\n\n
            yield f"event: {event}\ndata: {payload}\n\n"


def run_in_thread(target: Callable[[], None]) -> threading.Thread:
    """Start a daemon thread, copying current contextvars so ContextVar.get() works inside."""
    import contextvars

    ctx = contextvars.copy_context()

    def _runner() -> None:
        ctx.run(target)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t
