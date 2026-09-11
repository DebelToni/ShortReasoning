#!/usr/bin/env python3
"""Paired clean-history versus rewritten-history tool-reasoning experiments."""

from __future__ import annotations

import base64
import copy
import decimal
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = "https://openrouter.ai/api/v1/chat/completions"

TARGET_SYSTEM_TEMPLATE = """You are a careful decision analyst operating through function tools.
Use exactly one tool call per assistant turn until all evidence phases are complete.
Use the evidence phases in this exact order: {phase_order}.
Do not skip, combine, or predict a tool result. Do not invent missing values.
The exception tool is forbidden unless an authoritative result explicitly permits a waiver.
After the final evidence result, make the decision required by the authoritative rule and end with an exact line of the form FINAL: <candidate_id>.
Reasoning style is your choice; correctness and faithful state tracking are required."""

REWRITER_SYSTEM = """Rewrite an assistant reasoning scratchpad into a shorter state record.
Output only the rewritten scratchpad, with no preamble or code fence.
Preserve every decision-relevant fact, number, uncertainty, conclusion, and requested next action.
Preserve evidence-status distinctions exactly, including estimate versus binding quote and observed versus pending.
The requested tool call and its arguments are preserved separately outside the scratchpad. Treat them only as context: retain the intended next action, but never import facts, code, or argument details absent from the raw reasoning.
Aim for at most half the raw word count when the state permits, and never make the scratchpad longer.
Preserve the raw notation rather than translating mathematical expressions into implementation syntax or otherwise improving them.
Do not add knowledge, improve the solution, or predict the requested tool's result: that tool has not run yet.
Remove restatement, headings, rhetorical transitions, and repeated arithmetic. Fragments, equations, and compact clauses are allowed."""


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in {'"', "'"}:
            quote = value[0]
            closing = value.find(quote, 1)
            if closing < 0:
                raise ValueError(f"unterminated quoted value for {name}")
            value = value[1:closing]
        else:
            value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(name, value)


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        value,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def openrouter_key(variable: str = "OPENROUTER_API_KEY") -> str:
    key = os.environ.get(variable, "")
    if not key:
        raise RuntimeError(f"{variable} is missing")
    # The local file currently contains a prefix typo; never write the corrected secret.
    if key.startswith("sk-or-v2-"):
        key = "sk-or-v1-" + key.removeprefix("sk-or-v2-")
    return key


class BudgetFloorReached(RuntimeError):
    """Raised before a paid OpenRouter call would violate the configured balance floor."""


class ChatAttemptError(RuntimeError):
    """One journalable OpenRouter transport/HTTP attempt failed."""

    def __init__(
        self,
        message: str,
        *,
        wall_seconds: float,
        http_status: int | None = None,
        response_body: str | None = None,
        response_body_bytes: bytes | None = None,
    ):
        super().__init__(message)
        self.wall_seconds = wall_seconds
        self.http_status = http_status
        self.response_body = response_body
        self.response_body_bytes = response_body_bytes

    def record(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "wall_seconds": self.wall_seconds,
            "http_status": self.http_status,
            "response_body": self.response_body,
            "response_body_base64": (
                base64.b64encode(self.response_body_bytes).decode("ascii")
                if self.response_body_bytes is not None
                else None
            ),
        }


_BUDGET_LOCK = threading.Lock()
_PAID_CALL_LOCK = threading.Lock()
_BUDGET_CHECKED_AT = 0.0
_BUDGET_REMAINING: float | None = None


def remaining_openrouter_credits(
    key_variable: str = "OPENROUTER_API_KEY",
) -> float:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={
            "Authorization": f"Bearer {openrouter_key(key_variable)}",
            "User-Agent": "ShortReasoning budget guard",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read())["data"]
    return float(data["total_credits"]) - float(data["total_usage"])


def enforce_openrouter_budget(payload: dict[str, Any], *, max_cache_seconds: float = 15.0) -> float | None:
    floor_text = os.environ.get("OPENROUTER_MIN_BALANCE_USD")
    model = str(payload.get("model", ""))
    if not floor_text or model.endswith(":free"):
        return None
    floor = float(floor_text)
    reserve = float(os.environ.get("OPENROUTER_MAX_CALL_COST_USD", "0"))
    global _BUDGET_CHECKED_AT, _BUDGET_REMAINING
    with _BUDGET_LOCK:
        now = time.monotonic()
        if _BUDGET_REMAINING is None or now - _BUDGET_CHECKED_AT >= max_cache_seconds:
            _BUDGET_REMAINING = remaining_openrouter_credits()
            _BUDGET_CHECKED_AT = now
        remaining = _BUDGET_REMAINING
    if remaining - reserve <= floor:
        raise BudgetFloorReached(
            f"OpenRouter balance ${remaining:.4f} cannot preserve the configured ${floor:.2f} "
            f"floor plus ${reserve:.2f} maximum-call reserve"
        )
    return remaining


def post_chat_once_raw(payload: dict[str, Any]) -> tuple[bytes, float]:
    """Make exactly one guarded request and return its unparsed HTTP body."""
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {openrouter_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.short-reasoning/",
            "X-Title": "ShortReasoning experiment",
        },
        method="POST",
    )
    paid = bool(os.environ.get("OPENROUTER_MIN_BALANCE_USD")) and not str(
        payload.get("model", "")
    ).endswith(":free")
    started = time.perf_counter()
    try:
        if paid:
            with _PAID_CALL_LOCK:
                enforce_openrouter_budget(payload, max_cache_seconds=0)
                with urllib.request.urlopen(request, timeout=600) as response:
                    result = response.read()
        else:
            enforce_openrouter_budget(payload)
            with urllib.request.urlopen(request, timeout=600) as response:
                result = response.read()
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        body_text = body_bytes.decode(errors="replace")
        exc.close()
        wall = time.perf_counter() - started
        raise ChatAttemptError(
            f"OpenRouter HTTP {exc.code}: {body_text}",
            wall_seconds=wall,
            http_status=exc.code,
            response_body=body_text,
            response_body_bytes=body_bytes,
        ) from exc
    except Exception as exc:
        if isinstance(exc, BudgetFloorReached):
            raise
        wall = time.perf_counter() - started
        raise ChatAttemptError(
            f"{type(exc).__name__}: {exc}", wall_seconds=wall
        ) from exc
    return result, time.perf_counter() - started


def post_chat_once(payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Compatibility wrapper for callers that do not need response-first journaling."""
    raw_body, wall = post_chat_once_raw(payload)
    return strict_json_loads(raw_body), wall


def post_chat(payload: dict[str, Any], max_attempts: int = 6) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    paid_guard_enabled = bool(os.environ.get("OPENROUTER_MIN_BALANCE_USD")) and not str(
        payload.get("model", "")
    ).endswith(":free")

    def request_once(request: urllib.request.Request) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read())
        return result, time.perf_counter() - started

    for attempt in range(max_attempts):
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {openrouter_key()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local.short-reasoning/",
                "X-Title": "ShortReasoning experiment",
            },
            method="POST",
        )
        try:
            if paid_guard_enabled:
                with _PAID_CALL_LOCK:
                    enforce_openrouter_budget(payload, max_cache_seconds=0)
                    result, wall = request_once(request)
            else:
                enforce_openrouter_budget(payload)
                result, wall = request_once(request)
            if result.get("choices"):
                return result, wall
            error = result.get("error", {})
            code = error.get("code")
            retryable = code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt + 1 == max_attempts:
                raise RuntimeError(f"OpenRouter response error: {json.dumps(result)}")
            time.sleep(min(15 * 2**attempt, 120) + random.random())
            continue
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode(errors="replace")
            retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt + 1 == max_attempts:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {error_body}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(15 * 2**attempt, 120)
            time.sleep(delay + random.random())
    raise AssertionError("unreachable")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = json.loads(path.read_text())
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for task in tasks:
        task_id = task["id"]
        if task_id in ids:
            raise ValueError(f"duplicate task id: {task_id}")
        ids.add(task_id)
        phases = task["phases"]
        if len(phases) < 4:
            raise ValueError(f"{task_id}: expected at least four phases")
        if not 0 < task["fork_after"] < len(phases):
            raise ValueError(f"{task_id}: invalid fork_after")
        names = [phase["name"] for phase in phases] + [task["distractor"]["name"]]
        if len(names) != len(set(names)):
            raise ValueError(f"{task_id}: duplicate tool name")
        for phase in phases:
            if phase["expected_arguments"].get("case_id") != task["case_id"]:
                raise ValueError(f"{task_id}: phase case_id mismatch")


def tool_schema(
    name: str,
    description: str,
    argument_list: str | None,
    argument_count: int | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {"case_id": {"type": "string"}}
    required = ["case_id"]
    if argument_list:
        array_schema: dict[str, Any] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": argument_count or 1,
            "uniqueItems": True,
        }
        if argument_count is not None:
            array_schema["maxItems"] = argument_count
        properties[argument_list] = array_schema
        required.append(argument_list)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def build_tools(task: dict[str, Any]) -> list[dict[str, Any]]:
    tools = [
        tool_schema(
            phase["name"],
            phase["description"],
            phase.get("argument_list"),
            len(phase["expected_arguments"][phase["argument_list"]])
            if phase.get("argument_list")
            else None,
        )
        for phase in task["phases"]
    ]
    tools.append(tool_schema(task["distractor"]["name"], task["distractor"]["description"], None))
    return tools


def initial_messages(task: dict[str, Any]) -> list[dict[str, Any]]:
    order = " -> ".join(phase["name"] for phase in task["phases"])
    return [
        {"role": "system", "content": TARGET_SYSTEM_TEMPLATE.format(phase_order=order)},
        {"role": "user", "content": f"Case ID: {task['case_id']}.\n{task['user_prompt']}"},
    ]


def reasoning_text(message: dict[str, Any]) -> str:
    if not isinstance(message, dict):
        return ""
    if message.get("reasoning") is not None:
        return str(message["reasoning"])
    if message.get("reasoning_content") is not None:
        return str(message["reasoning_content"])
    details = message.get("reasoning_details")
    if not isinstance(details, list):
        return ""
    return "\n".join(
        str(block.get("text", ""))
        for block in details
        if isinstance(block, dict) and block.get("text")
    )


def assistant_history_message(message: dict[str, Any], reasoning: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    reasoning_value = reasoning_text(message) if reasoning is None else reasoning
    if reasoning_value:
        result["reasoning"] = reasoning_value
    if message.get("tool_calls") is not None:
        result["tool_calls"] = copy.deepcopy(message["tool_calls"])
    return result


def target_payload(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model["model"],
        "messages": copy.deepcopy(messages),
        "tools": copy.deepcopy(tools),
        "tool_choice": "auto",
        "reasoning": {"effort": model["reasoning_effort"], "exclude": False},
        "include_reasoning": True,
        "temperature": 0,
        "max_tokens": model["max_tokens"],
        "provider": copy.deepcopy(model["provider"]),
    }
    if model.get("seed_supported", True):
        payload["seed"] = seed
    return payload


def rewrite_payload(rewriter: dict[str, Any], raw_reasoning: str, tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call["function"]
    user = {
        "reasoning": raw_reasoning,
        "requested_tool": {
            "name": function["name"],
            "arguments": function["arguments"],
        },
    }
    return {
        "model": rewriter["model"],
        "messages": [
            {"role": "system", "content": REWRITER_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "reasoning": {"effort": rewriter["reasoning_effort"], "exclude": True},
        "include_reasoning": False,
        "seed": 99173,
        "max_tokens": rewriter["max_tokens"],
        "provider": copy.deepcopy(rewriter["provider"]),
    }


def strip_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:text|markdown)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]*-?\d+\b")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])\$?(\d[\d,]*(?:\.\d+)?)(?:\s*(%|[kKmM])(?![A-Za-z]))?"
)


def normalized_numeric_literals(text: str) -> list[str]:
    text_without_ids = IDENTIFIER_RE.sub("", text.upper())
    values = set()
    for match in NUMBER_RE.finditer(text_without_ids):
        value = decimal.Decimal(match.group(1).replace(",", ""))
        suffix = match.group(2)
        if suffix == "%":
            value /= 100
        elif suffix and suffix.lower() == "k":
            value *= 1000
        elif suffix and suffix.lower() == "m":
            value *= 1000000
        values.add(format(value.normalize(), "f"))
    return sorted(values)


def rewrite_fidelity(raw_reasoning: str, rewritten: str) -> dict[str, Any]:
    raw_ids = sorted(set(IDENTIFIER_RE.findall(raw_reasoning.upper())))
    rewritten_ids = set(IDENTIFIER_RE.findall(rewritten.upper()))
    raw_numbers = normalized_numeric_literals(raw_reasoning)
    rewritten_numbers = set(normalized_numeric_literals(rewritten))
    missing_ids = [identifier for identifier in raw_ids if identifier not in rewritten_ids]
    missing_numbers = [number for number in raw_numbers if number not in rewritten_numbers]
    return {
        "raw_identifiers": raw_ids,
        "missing_identifiers": missing_ids,
        "all_identifiers_preserved": not missing_ids,
        "raw_numeric_literals": raw_numbers,
        "missing_numeric_literals": missing_numbers,
        "all_numeric_literals_preserved": not missing_numbers,
    }


def collect_task_candidate_ids(value: Any) -> set[str]:
    result = set()
    if isinstance(value, dict):
        if isinstance(value.get("id"), str):
            result.add(value["id"].upper())
        for child in value.values():
            result.update(collect_task_candidate_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(collect_task_candidate_ids(child))
    return result


def task_hard_thresholds(task: dict[str, Any]) -> set[str]:
    constraints = task["phases"][0]["result"]["hard_constraints"]
    return {
        format(decimal.Decimal(str(value)).normalize(), "f")
        for value in constraints.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


THRESHOLD_CONTEXT_RE = re.compile(
    r"constraint|hard|lte|gte|[<>≤≥]|maximum|minimum|max|min|limit|cap|threshold|"
    r"binding|quote|cost|expected|probability|rate|loss|power|downlink|"
    r"units|people|route|facility|independent|station|region|day",
    re.IGNORECASE,
)


def hard_thresholds_in_reasoning(raw_reasoning: str, thresholds: set[str]) -> list[str]:
    without_list_numbers = re.sub(
        r"(?m)^\s*\d+[.)]\s+", "", raw_reasoning.upper()
    )
    without_phase_numbers = re.sub(
        r"\bPHASE\s*[-:]?\s*\d+\b", "PHASE", without_list_numbers
    )
    text_without_ids = IDENTIFIER_RE.sub("", without_phase_numbers)
    found = set()
    for match in NUMBER_RE.finditer(text_without_ids):
        value = decimal.Decimal(match.group(1).replace(",", ""))
        suffix = match.group(2)
        if suffix == "%":
            value /= 100
        elif suffix and suffix.lower() == "k":
            value *= 1000
        elif suffix and suffix.lower() == "m":
            value *= 1000000
        normalized = format(value.normalize(), "f")
        context = text_without_ids[max(0, match.start() - 64):match.end() + 64]
        if normalized in thresholds and THRESHOLD_CONTEXT_RE.search(context):
            found.add(normalized)
    return sorted(found)


def rewrite_before_tool_result(
    rewriter: dict[str, Any],
    raw_reasoning: str,
    tool_call: dict[str, Any],
    required_candidate_ids: list[str] | None = None,
    required_hard_thresholds: list[str] | None = None,
    post_chat_fn: Any = None,
) -> dict[str, Any]:
    required_candidate_ids = required_candidate_ids or []
    required_hard_thresholds = required_hard_thresholds or []
    payload = rewrite_payload(rewriter, raw_reasoning, tool_call)
    attempts = []
    chat = post_chat if post_chat_fn is None else post_chat_fn
    for _ in range(3):
        response, wall = chat(payload)
        attempts.append({
            "request": copy.deepcopy(payload),
            "response": response,
            "wall_seconds": wall,
        })
        message = response["choices"][0]["message"]
        rewritten = strip_fence(message.get("content") or "")
        if not rewritten:
            payload = copy.deepcopy(payload)
            payload["max_tokens"] = min(payload["max_tokens"] * 2, 8192)
            continue

        rewritten_ids = set(IDENTIFIER_RE.findall(rewritten.upper()))
        rewritten_numbers = set(normalized_numeric_literals(rewritten))
        missing_candidates = [item for item in required_candidate_ids if item not in rewritten_ids]
        missing_thresholds = [item for item in required_hard_thresholds if item not in rewritten_numbers]
        fidelity = rewrite_fidelity(raw_reasoning, rewritten)
        fidelity.update({
            "required_candidate_ids": required_candidate_ids,
            "missing_candidate_ids": missing_candidates,
            "all_candidate_ids_preserved": not missing_candidates,
            "required_hard_thresholds": required_hard_thresholds,
            "missing_hard_thresholds": missing_thresholds,
            "all_hard_thresholds_preserved": not missing_thresholds,
        })
        if not missing_candidates and not missing_thresholds:
            return {
                "attempts": attempts,
                "request": copy.deepcopy(payload),
                "response": response,
                "wall_seconds": wall,
                "raw_reasoning": raw_reasoning,
                "rewritten": rewritten,
                "raw_words": len(raw_reasoning.split()),
                "rewritten_words": len(rewritten.split()),
                "requested_tool": copy.deepcopy(tool_call["function"]),
                "fidelity": fidelity,
            }

        repair = {
            "reasoning": raw_reasoning,
            "requested_tool": copy.deepcopy(tool_call["function"]),
            "prior_draft": rewritten,
            "required_candidate_ids": required_candidate_ids,
            "required_hard_thresholds": required_hard_thresholds,
            "repair_instruction": (
                "Rewrite again. Integrate every required candidate ID and hard-threshold value "
                "semantically into the compact state; do not append an unexplained token list."
            ),
        }
        payload = rewrite_payload(rewriter, raw_reasoning, tool_call)
        payload["messages"][1]["content"] = json.dumps(repair, ensure_ascii=False)
    raise RuntimeError("rewriter failed the candidate/threshold fidelity check three times")


def parse_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    arguments = tool_call["function"].get("arguments", "{}")
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise TypeError("tool arguments must be an object or JSON string")

    return strict_json_loads(arguments)


def arguments_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, list):
            if not isinstance(actual_value, list) or len(actual_value) != len(expected_value):
                return False
            actual_items = sorted(
                json.dumps(item, ensure_ascii=False, sort_keys=True) for item in actual_value
            )
            expected_items = sorted(
                json.dumps(item, ensure_ascii=False, sort_keys=True) for item in expected_value
            )
            if actual_items != expected_items:
                return False
        elif actual_value != expected_value:
            return False
    return True


def evaluate_tool_response(message: dict[str, Any], phase: dict[str, Any]) -> dict[str, Any]:
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    result: dict[str, Any] = {
        "expected_tool": phase["name"],
        "expected_arguments": phase["expected_arguments"],
        "call_count": len(calls) if isinstance(calls, list) else 0,
        "correct": False,
    }
    if calls is None:
        return result
    if not isinstance(calls, list):
        result["shape_error"] = "tool_calls must be a list"
        return result
    if len(calls) != 1:
        return result
    try:
        call = calls[0]
        if not isinstance(call, dict):
            raise TypeError("tool call must be an object")
        if call.get("type") != "function":
            raise TypeError("tool call type must be function")
        if not isinstance(call.get("id"), str) or not call["id"]:
            raise TypeError("tool call id must be a nonempty string")
        function = call["function"]
        if not isinstance(function, dict):
            raise TypeError("tool function must be an object")
        if not isinstance(function.get("name"), str) or not function["name"]:
            raise TypeError("tool function name must be a nonempty string")
        result["actual_tool"] = function["name"]
        actual_arguments = parse_arguments(call)
        if not isinstance(actual_arguments, dict):
            raise TypeError("tool arguments must decode to an object")
        result["actual_arguments"] = actual_arguments
        result["correct"] = (
            result["actual_tool"] == phase["name"]
            and arguments_match(actual_arguments, phase["expected_arguments"])
        )
    except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result["argument_error"] = str(exc)
    return result


def response_metrics(response: dict[str, Any], wall: float) -> dict[str, Any]:
    choices = response.get("choices") if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    reasoning = reasoning_text(message)
    return {
        "provider": response.get("provider") if isinstance(response, dict) else None,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "reasoning_chars": len(reasoning),
        "reasoning_words": len(reasoning.split()),
        "cost": usage.get("cost", 0),
        "wall_seconds": wall,
    }


def call_target(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    seed: int,
    post_chat_fn: Any = None,
) -> dict[str, Any]:
    payload = target_payload(model, messages, tools, seed)
    chat = post_chat if post_chat_fn is None else post_chat_fn
    response, wall = chat(payload)
    return {
        "request": copy.deepcopy(payload),
        "response": response,
        "wall_seconds": wall,
        "metrics": response_metrics(response, wall),
    }


def append_tool_cycle(
    messages: list[dict[str, Any]],
    message: dict[str, Any],
    reasoning: str,
    phase: dict[str, Any],
) -> None:
    call = message["tool_calls"][0]
    messages.append(assistant_history_message(message, reasoning))
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": phase["name"],
            "content": json.dumps(phase["result"], ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        }
    )


def final_is_correct(content: str | None, expected: str) -> bool:
    if not isinstance(content, str) or not content:
        return False
    nonempty_lines = [line for line in content.splitlines() if line.strip()]
    final_declarations = [
        line
        for line in nonempty_lines
        if re.match(r"(?i)^\s*(?:\*\*)?FINAL:", line)
    ]
    if len(final_declarations) != 1 or final_declarations[0] != nonempty_lines[-1]:
        return False
    return bool(
        re.fullmatch(
            rf"(?i)\s*(?:\*\*)?FINAL:\s*{re.escape(expected)}(?:\*\*)?\s*",
            final_declarations[0],
        )
    )


def run_case(
    task: dict[str, Any],
    model_name: str,
    model: dict[str, Any],
    rewriter: dict[str, Any],
    base_seed: int,
) -> dict[str, Any]:
    tools = build_tools(task)
    clean_messages = initial_messages(task)
    rewritten_messages = copy.deepcopy(clean_messages)
    shared_records: list[dict[str, Any]] = []
    task_candidate_ids = collect_task_candidate_ids(task)
    hard_thresholds = task_hard_thresholds(task)

    for phase_index, phase in enumerate(task["phases"][: task["fork_after"]]):
        target = call_target(model, clean_messages, tools, base_seed + phase_index)
        message = target["response"]["choices"][0]["message"]
        action = evaluate_tool_response(message, phase)
        target["turn"] = phase_index + 1
        target["phase"] = phase["name"]
        target["action"] = action
        if not action["correct"]:
            return {
                "task_id": task["id"],
                "model_name": model_name,
                "model": model,
                "base_seed": base_seed,
                "status": "shared_action_failure",
                "shared": shared_records + [target],
            }

        raw = reasoning_text(message)
        required_candidate_ids = sorted(
            set(IDENTIFIER_RE.findall(raw.upper())) & task_candidate_ids
        )
        required_hard_thresholds = hard_thresholds_in_reasoning(raw, hard_thresholds)
        rewrite = rewrite_before_tool_result(
            rewriter,
            raw,
            message["tool_calls"][0],
            required_candidate_ids,
            required_hard_thresholds,
        )
        # The rewrite request above is complete before this result is appended anywhere.
        append_tool_cycle(clean_messages, message, raw, phase)
        append_tool_cycle(rewritten_messages, message, rewrite["rewritten"], phase)
        target["rewrite"] = rewrite
        shared_records.append(target)

    branches: dict[str, dict[str, Any]] = {
        "clean": {"messages": clean_messages, "turns": [], "actions_correct": True},
        "rewritten": {"messages": rewritten_messages, "turns": [], "actions_correct": True},
    }

    for phase_index in range(task["fork_after"], len(task["phases"])):
        phase = task["phases"][phase_index]
        order = ["clean", "rewritten"]
        random.Random(base_seed + phase_index).shuffle(order)
        for condition in order:
            branch = branches[condition]
            if branch.get("status") == "action_failure":
                continue
            target = call_target(model, branch["messages"], tools, base_seed + phase_index)
            message = target["response"]["choices"][0]["message"]
            action = evaluate_tool_response(message, phase)
            target["turn"] = phase_index + 1
            target["phase"] = phase["name"]
            target["action"] = action
            branch["turns"].append(target)
            branch["actions_correct"] = branch["actions_correct"] and action["correct"]
            if action["correct"]:
                append_tool_cycle(branch["messages"], message, reasoning_text(message), phase)
            else:
                branch["status"] = "action_failure"

    final_turn = len(task["phases"]) + 1
    order = ["clean", "rewritten"]
    random.Random(base_seed + final_turn).shuffle(order)
    for condition in order:
        branch = branches[condition]
        if branch.get("status") == "action_failure":
            continue
        target = call_target(model, branch["messages"], tools, base_seed + final_turn)
        message = target["response"]["choices"][0]["message"]
        content = message.get("content")
        target["turn"] = final_turn
        target["phase"] = "final"
        target["final_correct"] = final_is_correct(content, task["final_answer"])
        target["unexpected_tool_calls"] = len(message.get("tool_calls") or [])
        branch["turns"].append(target)
        branch["final_correct"] = target["final_correct"] and target["unexpected_tool_calls"] == 0
        branch["final_content"] = content
        branch["status"] = "complete"

    for branch in branches.values():
        branch.pop("messages", None)
        metrics = [turn["metrics"] for turn in branch["turns"]]
        branch["aggregate"] = aggregate_metrics(metrics)
        branch.setdefault("final_correct", False)
        branch["success"] = branch["actions_correct"] and branch["final_correct"]

    clean = branches["clean"]["aggregate"]
    rewritten = branches["rewritten"]["aggregate"]
    comparable = (
        branches["clean"].get("status") == "complete"
        and branches["rewritten"].get("status") == "complete"
        and clean["turn_count"] == rewritten["turn_count"]
    )
    clean_first = branches["clean"]["turns"][0]["metrics"]
    rewritten_first = branches["rewritten"]["turns"][0]["metrics"]
    pair = {
        "comparable_horizon": comparable,
        "reasoning_token_change_percent": percent_change(
            clean.get("reasoning_tokens"), rewritten.get("reasoning_tokens")
        ) if comparable else None,
        "completion_token_change_percent": percent_change(
            clean.get("completion_tokens"), rewritten.get("completion_tokens")
        ) if comparable else None,
        "reasoning_word_change_percent": percent_change(
            clean.get("reasoning_words"), rewritten.get("reasoning_words")
        ) if comparable else None,
        "prompt_token_change_percent": percent_change(
            clean.get("prompt_tokens"), rewritten.get("prompt_tokens")
        ) if comparable else None,
        "first_postfork_reasoning_token_change_percent": percent_change(
            clean_first.get("reasoning_tokens"), rewritten_first.get("reasoning_tokens")
        ),
    }
    return {
        "task_id": task["id"],
        "title": task["title"],
        "model_name": model_name,
        "model": model,
        "base_seed": base_seed,
        "status": "complete",
        "fork_after": task["fork_after"],
        "expected_final": task["final_answer"],
        "shared": shared_records,
        "branches": branches,
        "pair": pair,
    }


def percent_change(clean: int | float | None, rewritten: int | float | None) -> float | None:
    if clean in (None, 0) or rewritten is None:
        return None
    return (rewritten - clean) / clean * 100


def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"turn_count": len(metrics)}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "reasoning_chars",
        "reasoning_words",
        "cost",
        "wall_seconds",
    ):
        values = [metric.get(key) for metric in metrics]
        result[key] = None if any(value is None for value in values) else sum(values)
        result[f"{key}_by_turn"] = values
    return result


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [case for case in cases if case.get("status") == "complete"]
    by_model: dict[str, Any] = {}
    for model_name in sorted({case["model_name"] for case in cases}):
        model_cases = [case for case in complete if case["model_name"] == model_name]
        effects = [
            case["pair"]["reasoning_token_change_percent"]
            for case in model_cases
            if case["pair"]["reasoning_token_change_percent"] is not None
        ]
        first_turn_effects = [
            case["pair"]["first_postfork_reasoning_token_change_percent"]
            for case in model_cases
            if case["pair"]["first_postfork_reasoning_token_change_percent"] is not None
        ]
        completion_effects = [
            case["pair"]["completion_token_change_percent"]
            for case in model_cases
            if case["pair"]["completion_token_change_percent"] is not None
        ]
        prompt_effects = [
            case["pair"]["prompt_token_change_percent"]
            for case in model_cases
            if case["pair"]["prompt_token_change_percent"] is not None
        ]
        by_model[model_name] = {
            "cases_complete": len(model_cases),
            "clean_successes": sum(case["branches"]["clean"]["success"] for case in model_cases),
            "rewritten_successes": sum(case["branches"]["rewritten"]["success"] for case in model_cases),
            "clean_action_failures": sum(case["branches"]["clean"].get("status") == "action_failure" for case in model_cases),
            "rewritten_action_failures": sum(case["branches"]["rewritten"].get("status") == "action_failure" for case in model_cases),
            "clean_final_tool_calls": sum(
                bool(case["branches"]["clean"]["turns"][-1].get("unexpected_tool_calls"))
                for case in model_cases
                if case["branches"]["clean"]["turns"]
            ),
            "rewritten_final_tool_calls": sum(
                bool(case["branches"]["rewritten"]["turns"][-1].get("unexpected_tool_calls"))
                for case in model_cases
                if case["branches"]["rewritten"]["turns"]
            ),
            "comparable_horizon_cases": sum(case["pair"]["comparable_horizon"] for case in model_cases),
            "reasoning_reduced_cases": sum(effect < 0 for effect in effects),
            "reasoning_increased_cases": sum(effect > 0 for effect in effects),
            "reasoning_token_changes_percent": effects,
            "median_reasoning_token_change_percent": statistics.median(effects) if effects else None,
            "median_first_postfork_reasoning_token_change_percent": statistics.median(first_turn_effects) if first_turn_effects else None,
            "median_completion_token_change_percent": statistics.median(completion_effects) if completion_effects else None,
            "median_prompt_token_change_percent": statistics.median(prompt_effects) if prompt_effects else None,
        }
    all_effects = [
        case["pair"]["reasoning_token_change_percent"]
        for case in complete
        if case["pair"]["reasoning_token_change_percent"] is not None
    ]
    return {
        "cases_total": len(cases),
        "cases_complete": len(complete),
        "comparable_horizon_cases": len(all_effects),
        "reasoning_reduced_cases": sum(effect < 0 for effect in all_effects),
        "reasoning_increased_cases": sum(effect > 0 for effect in all_effects),
        "by_model": by_model,
        "all_reasoning_token_changes_percent": all_effects,
        "median_reasoning_token_change_percent": statistics.median(all_effects) if all_effects else None,
    }


def render_summary(summary: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    lines = [
        "# ShortReasoning run summary",
        "",
        "Only clean plaintext history and blindly rewritten plaintext history were compared.",
        "",
        "| Model | Task | Seed | Clean outcome | Rewritten outcome | Clean reasoning by turn | Rewritten reasoning by turn | Cumulative change |",
        "|---|---|---:|---|---|---:|---:|---:|", 
    ]
    for case in cases:
        if case.get("status") != "complete":
            lines.append(
                f"| {case['model_name']} | {case['task_id']} | {case.get('base_seed', '—')} | — | — | — | — | {case['status']} |"
            )
            continue
        clean = case["branches"]["clean"]
        rewritten = case["branches"]["rewritten"]
        change = case["pair"]["reasoning_token_change_percent"]
        change_text = "n/a" if change is None else f"{change:+.1f}%"
        def outcome(branch):
            if branch["success"]:
                return "success"
            if branch.get("status") == "action_failure":
                return "tool-action failure"
            if branch.get("status") == "complete" and not branch["final_correct"]:
                final_turn = branch["turns"][-1]
                if final_turn.get("unexpected_tool_calls"):
                    return "unexpected final tool"
                return "wrong/invalid final"
            return branch.get("status", "failed")

        lines.append(
            f"| {case['model_name']} | {case['task_id']} | {case['base_seed']} | {outcome(clean)} | "
            f"{outcome(rewritten)} | {clean['aggregate']['reasoning_tokens_by_turn']} | "
            f"{rewritten['aggregate']['reasoning_tokens_by_turn']} | {change_text} |"
        )
    lines.extend(["", "## Findings", ""])
    for model_name, model_summary in summary["by_model"].items():
        median = model_summary["median_reasoning_token_change_percent"]
        median_text = "n/a" if median is None else f"{median:+.1f}%"
        lines.append(
            f"- **{model_name}:** reasoning fell in {model_summary['reasoning_reduced_cases']}/"
            f"{model_summary['comparable_horizon_cases']} comparable tasks; median change {median_text}; "
            f"clean success {model_summary['clean_successes']}/{model_summary['cases_complete']}, "
            f"rewritten success {model_summary['rewritten_successes']}/{model_summary['cases_complete']}."
        )
    lines.extend([
        "",
        "Final failures distinguish an unexpected tool call from a wrong or incorrectly formatted final answer. "
        "Token changes are reported only when both branches reached the same number of post-fork turns.",
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ])
    return "\n".join(lines)
