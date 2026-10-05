import json
import os
import time
from typing import Any, Callable
from config import LOG_FILE
from langchain_core.callbacks import BaseCallbackHandler
#provides callback methods that LangChain can trigger when things happen during an LLM call.

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log_event(heal_log: list[dict], step: str, status: str, detail: str = "") -> None:
    """
    status: ok | info | problem | recovery | recovered | failed
    Stored in the list (shown in the UI) and appended to a JSONL file (permanent).
    """
    event = {"time": time.strftime("%H:%M:%S"), "step": step, "status": status, "detail": detail}
    heal_log.append(event)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def run_with_healing(
    name: str,
    task: Callable[[Any], Any],
    validate: Callable[[Any], tuple[bool, str]],
    recover: Callable[[Any, str, Any], Any],
    max_retries: int,
    heal_log: list[dict],
    arg: Any,
) -> tuple[Any, bool]:
    """
    EXECUTE -> VALIDATE -> DIAGNOSE+RECOVER -> RETRY (bounded by max_retries).
    Every outcome is logged. Returns (last_result, succeeded).
    """
    result = None
    for attempt in range(max_retries + 1):
        try:
            result = task(arg)
            ok, problem = validate(result)
        except Exception as e:                       # a crash is just another failure
            result, ok, problem = None, False, f"exception: {e}"

        if ok:
            log_event(heal_log, name, "recovered" if attempt else "ok", str(arg)[:150])
            return result, True

        log_event(heal_log, name, "problem", problem[:200])

        if attempt == max_retries:
            log_event(heal_log, name, "failed", "retries exhausted")
            break

        new_arg = recover(arg, problem, result)
        if new_arg is None:
            log_event(heal_log, name, "failed", "no recovery possible")
            break
        log_event(heal_log, name, "recovery", f"retry {attempt + 1}/{max_retries} with: {str(new_arg)[:150]}")
        arg = new_arg

    return result, False

# ---------- LLM usage logging (tokens + latency) ----------
LLM_CALLS: list[dict] = []      # in-memory, for the UI / summary
_start_times: dict = {}


class UsageLogger(BaseCallbackHandler):
    """Attached once in config.get_llm(); logs every LLM call automatically."""

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        _start_times[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs):
        seconds = time.perf_counter() - _start_times.pop(run_id, time.perf_counter())
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            usage = {}
        self._record("ok", seconds, usage)

    def on_llm_error(self, error, *, run_id, **kwargs):
        seconds = time.perf_counter() - _start_times.pop(run_id, time.perf_counter())
        self._record("error", seconds, {}, str(error)[:150])

    @staticmethod
    def _record(status: str, seconds: float, usage: dict, error: str = "") -> None:
        call = {
            "time": time.strftime("%H:%M:%S"), "step": "llm", "status": status,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "seconds": round(seconds, 2), "error": error,
        }
        LLM_CALLS.append(call)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(call) + "\n")


def usage_summary() -> dict:
    return {
        "llm_calls": len(LLM_CALLS),
        "input_tokens": sum(c["input_tokens"] for c in LLM_CALLS),
        "output_tokens": sum(c["output_tokens"] for c in LLM_CALLS),
        "llm_seconds": round(sum(c["seconds"] for c in LLM_CALLS), 2),
    }