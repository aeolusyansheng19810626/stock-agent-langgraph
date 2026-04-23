import json
import os
import uuid
import contextlib
from datetime import datetime

HISTORY_FILE = os.getenv("HISTORY_FILE", "history.json")

try:
    from filelock import FileLock as _FileLock
    def _lock():
        return _FileLock(HISTORY_FILE + ".lock")
except ImportError:
    @contextlib.contextmanager
    def _noop():
        yield
    def _lock():
        return _noop()


def save_history(record: dict) -> None:
    """Append one record to history.json; creates the file if absent."""
    try:
        with _lock():
            records = _read_raw()
            records.append(record)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # history write failure must never crash the main flow


def load_history() -> list[dict]:
    """Return all records newest-first."""
    return list(reversed(_read_raw()))


def clear_history() -> None:
    """Truncate history file."""
    try:
        with _lock():
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
    except Exception:
        pass


def _read_raw() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


# ── helpers used by app.py ────────────────────────────────────────────────────

def extract_nodes_called(tool_calls: list) -> list[str]:
    """Return unique node names in call order from tool_calls list."""
    seen: list[str] = []
    for tc in tool_calls:
        node = tc.get("node") or tc.get("tool_args", {}).get("node")
        if node and node not in seen:
            seen.append(node)
    return seen


def aggregate_token_usage(tool_calls: list) -> dict | None:
    """Sum token_usage across all tool_call entries; None if none found."""
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    found = False
    for tc in tool_calls:
        usage = tc.get("token_usage")
        if usage:
            found = True
            total["prompt_tokens"]     += usage.get("prompt_tokens", 0)
            total["completion_tokens"] += usage.get("completion_tokens", 0)
            total["total_tokens"]      += usage.get("total_tokens", 0)
    return total if found else None


def make_record(
    user_input: str,
    tool_calls: list,
    final_model: str,
    elapsed: float,
    has_error: bool,
    tickers: list | None = None,
) -> dict:
    return {
        "id":               str(uuid.uuid4()),
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input":       user_input,
        "tickers":          tickers or [],
        "nodes_called":     extract_nodes_called(tool_calls),
        "elapsed_seconds":  elapsed,
        "token_usage":      aggregate_token_usage(tool_calls),
        "final_model":      final_model,
        "has_error":        has_error,
    }
